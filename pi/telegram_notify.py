"""
Project Qwen - security push notifications via a Telegram bot (with photos).

Credentials live in .env (gitignored, chmod 600) - never in code:
    TELEGRAM_BOT_TOKEN=123456789:AA...
    TELEGRAM_CHAT_ID=987654321          # filled in by --setup

Sending never blocks the caller: notify() queues the message and a background
thread delivers it, retrying with backoff if the internet is down (queue
capped at MAX_QUEUE; oldest dropped first). So recording and the alarm are
never held up by Wi-Fi.

Photos/videos sent from files (send_photo_file / notify_video) are deleted
from the Pi once Telegram confirms delivery (delete_after=True) - the Pi
keeps no media it doesn't need. A file that can't be sent (no bot set up,
too large, dropped from a full queue) is kept.

Setup (once, after creating the bot with @BotFather and messaging it "hi"):
    python telegram_notify.py --setup     # finds your chat id, saves it to .env
Test:
    python telegram_notify.py --test      # sends a text + a camera photo
"""

import os
import queue
import threading
import time

ENV_PATH = ".env"
API = "https://api.telegram.org/bot{token}/{method}"
REQUEST_TIMEOUT_S = 15
MAX_QUEUE = 50
RETRY_DELAYS_S = (2, 5, 15, 30, 60)      # then keeps retrying every 60 s
JPEG_QUALITY = 80
PHOTO_MAX_WIDTH = 1280
VIDEO_MAX_BYTES = 49 * 1024 * 1024     # Bot API upload limit is 50 MB
VIDEO_BITRATE = "1200k"


def load_env(path: str = ENV_PATH) -> dict:
    env = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    env[key.strip()] = value.strip()
    return env


def _save_env_value(key: str, value: str, path: str = ENV_PATH):
    lines = []
    if os.path.exists(path):
        with open(path) as f:
            lines = [l for l in f.read().splitlines() if not l.startswith(f"{key}=")]
    lines.append(f"{key}={value}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.chmod(path, 0o600)


class TelegramNotifier:
    def __init__(self, token: str | None = None, chat_id: str | None = None):
        env = load_env()
        self.token = token or env.get("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or env.get("TELEGRAM_CHAT_ID")
        self.enabled = bool(self.token and self.chat_id)
        self._queue: queue.Queue = queue.Queue(maxsize=MAX_QUEUE)
        self._stop_event = threading.Event()
        self._thread = None
        self.on_command = None          # callable(text) for messages from the owner's chat
        if not self.enabled:
            print("[telegram] not configured (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID in .env) "
                  "- notifications off")

    def start(self):
        if self.enabled and self._thread is None:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            threading.Thread(target=self._poll_commands, daemon=True).start()

    def _poll_commands(self):
        """Long-polls the bot for messages and passes the text of any that come from the
        owner's chat (TELEGRAM_CHAT_ID) to on_command. Anyone else is ignored."""
        import requests
        offset = None
        first = True
        while not self._stop_event.is_set():
            try:
                params = {"timeout": 0 if first else 25, "allowed_updates": '["message"]'}
                if offset is not None:
                    params["offset"] = offset
                r = requests.get(API.format(token=self.token, method="getUpdates"),
                                 params=params, timeout=35)
                updates = r.json().get("result", [])
            except Exception as e:
                print(f"[telegram] command poll failed: {e}")
                self._stop_event.wait(10)
                continue
            for u in updates:
                offset = u["update_id"] + 1
                if first:
                    continue                  # skip anything sent while the robot was off
                msg = u.get("message") or {}
                if str(msg.get("chat", {}).get("id")) != str(self.chat_id):
                    continue
                text = (msg.get("text") or "").strip()
                if text and self.on_command:
                    try:
                        self.on_command(text)
                    except Exception as e:
                        print(f"[telegram] command handler error: {e}")
            first = False

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def notify(self, text: str, frame=None):
        """Queues a message, with a photo if `frame` (BGR numpy image) is given.
        Never blocks."""
        if not self.enabled:
            return
        photo = None
        if frame is not None:
            try:
                import cv2
                h, w = frame.shape[:2]
                if w > PHOTO_MAX_WIDTH:
                    frame = cv2.resize(frame, (PHOTO_MAX_WIDTH, int(h * PHOTO_MAX_WIDTH / w)))
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                photo = buf.tobytes() if ok else None
            except Exception as e:
                print(f"[telegram] couldn't encode photo: {e}")
        stamped = f"{text}\n{time.strftime('%a %H:%M:%S')}"
        try:
            self._enqueue(("photo" if photo else "text", stamped, photo))
        except Exception as e:
            print(f"[telegram] couldn't queue message: {e}")

    def notify_video(self, path: str, caption: str, delete_after: bool = True):
        """Queues a recorded clip. It's converted to H.264 (plays inline on the
        iPhone) in the background before sending; the original is deleted once
        delivered (delete_after). Never blocks."""
        if self.enabled:
            self._enqueue(("video", f"{caption}\n{time.strftime('%a %H:%M:%S')}", (path, delete_after)))
        else:
            print(f"[telegram] not configured - keeping {path}")

    def send_photo_file(self, path: str, caption: str, delete_after: bool = True):
        """Queues a saved photo; deleted once delivered (delete_after). Never blocks."""
        if self.enabled:
            self._enqueue(("photo_file", f"{caption}\n{time.strftime('%a %H:%M:%S')}", (path, delete_after)))
        else:
            print(f"[telegram] not configured - keeping {path}")

    def _enqueue(self, item):
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            try:
                self._queue.get_nowait()          # drop the oldest
            except queue.Empty:
                pass
            self._queue.put_nowait(item)

    @staticmethod
    def _to_h264(path: str) -> str | None:
        """Converts a recording to an inline-playable H.264 mp4 (Pi hardware
        encoder first, libx264 fallback), at low CPU priority."""
        import subprocess
        out = path.rsplit(".", 1)[0] + ".telegram.mp4"
        for vcodec in (["-c:v", "h264_v4l2m2m", "-b:v", VIDEO_BITRATE],
                       ["-c:v", "libx264", "-preset", "veryfast", "-crf", "28"]):
            cmd = ["nice", "-n", "10", "ffmpeg", "-y", "-loglevel", "error", "-i", path,
                   *vcodec, "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "64k",
                   "-movflags", "+faststart", out]
            if subprocess.run(cmd, capture_output=True).returncode == 0 and os.path.getsize(out) > 0:
                return out
        return None

    @staticmethod
    def _delete(path: str):
        try:
            os.remove(path)
            print(f"[telegram] delivered - deleted {path}")
        except OSError as e:
            print(f"[telegram] couldn't delete {path}: {e}")

    def _send(self, kind: str, text: str, payload):
        import requests
        if kind == "photo_file":
            path, delete_after = payload
            with open(path, "rb") as f:
                r = requests.post(API.format(token=self.token, method="sendPhoto"),
                                  data={"chat_id": self.chat_id, "caption": text},
                                  files={"photo": ("qwen.jpg", f, "image/jpeg")},
                                  timeout=REQUEST_TIMEOUT_S * 2)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
            if delete_after:
                self._delete(path)
            return
        if kind == "video":
            path, delete_after = payload
            clip = self._to_h264(path)
            if clip is None:
                print(f"[telegram] couldn't convert {path} - sending a note instead")
                kind, text = "text", f"{text}\n(video kept on the robot: {path})"
            elif os.path.getsize(clip) > VIDEO_MAX_BYTES:
                os.remove(clip)
                kind, text = "text", f"{text}\n(video too large to send - kept on the robot: {path})"
            else:
                try:
                    with open(clip, "rb") as f:
                        r = requests.post(API.format(token=self.token, method="sendVideo"),
                                          data={"chat_id": self.chat_id, "caption": text,
                                                "supports_streaming": "true"},
                                          files={"video": ("qwen.mp4", f, "video/mp4")},
                                          timeout=120)
                finally:
                    os.remove(clip)
                if r.status_code != 200:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                if delete_after:
                    self._delete(path)
                return
        photo = payload if kind == "photo" else None
        if photo is None:
            r = requests.post(API.format(token=self.token, method="sendMessage"),
                              data={"chat_id": self.chat_id, "text": text},
                              timeout=REQUEST_TIMEOUT_S)
        else:
            r = requests.post(API.format(token=self.token, method="sendPhoto"),
                              data={"chat_id": self.chat_id, "caption": text},
                              files={"photo": ("qwen.jpg", photo, "image/jpeg")},
                              timeout=REQUEST_TIMEOUT_S)
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")

    def _run(self):
        while not self._stop_event.is_set():
            try:
                kind, text, payload = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            attempt = 0
            while not self._stop_event.is_set():
                try:
                    self._send(kind, text, payload)
                    print(f"[telegram] sent: {text.splitlines()[0]}"
                          + {"photo": " (+photo)", "photo_file": " (+photo)",
                             "video": " (+video)"}.get(kind, ""))
                    break
                except Exception as e:
                    delay = RETRY_DELAYS_S[min(attempt, len(RETRY_DELAYS_S) - 1)]
                    # Don't print the request URL - it contains the bot token.
                    print(f"[telegram] send failed ({type(e).__name__}), retrying in {delay}s")
                    attempt += 1
                    self._stop_event.wait(delay)


def _setup():
    import requests
    token = load_env().get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("No TELEGRAM_BOT_TOKEN in .env - add it first.")
    me = requests.get(API.format(token=token, method="getMe"), timeout=REQUEST_TIMEOUT_S).json()
    if not me.get("ok"):
        raise SystemExit("Telegram rejected the token - check it in .env.")
    print(f"Bot: @{me['result']['username']}")
    updates = requests.get(API.format(token=token, method="getUpdates"),
                           timeout=REQUEST_TIMEOUT_S).json().get("result", [])
    chats = {}
    for u in updates:
        msg = u.get("message") or u.get("edited_message")
        if msg and msg["chat"]["type"] == "private":
            chats[msg["chat"]["id"]] = msg["chat"].get("first_name", "")
    if not chats:
        raise SystemExit("No messages found - send your bot any message (e.g. 'hi') in Telegram, "
                         "then run --setup again.")
    if len(chats) > 1:
        raise SystemExit(f"Several people messaged the bot ({list(chats.values())}) - "
                         f"set TELEGRAM_CHAT_ID in .env by hand.")
    chat_id, name = next(iter(chats.items()))
    _save_env_value("TELEGRAM_CHAT_ID", str(chat_id))
    print(f"Chat found ({name}) - saved TELEGRAM_CHAT_ID to .env")


def _test():
    n = TelegramNotifier()
    if not n.enabled:
        raise SystemExit("Not configured - run --setup first.")
    n.start()
    n.notify("Qwen test: notifications are working.")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        for _ in range(15):
            ok, frame = cap.read()
        cap.release()
        if ok:
            n.notify("Qwen test: this is what my camera sees.", frame)
    except Exception as e:
        print(f"(no camera photo: {e})")
    time.sleep(8)
    n.stop()


if __name__ == "__main__":
    import sys
    if "--setup" in sys.argv:
        _setup()
    elif "--test" in sys.argv:
        _test()
    else:
        print(__doc__)
