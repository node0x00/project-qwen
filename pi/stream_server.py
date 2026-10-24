"""
Project Qwen - live computer-vision debug stream (MJPEG over HTTP).

Open on the Mac (browser, or VS Code: Cmd+Shift+P -> "Simple Browser: Show"):
    http://raspberrypi.local:8080/?key=<QWEN_STREAM_KEY>
The key is generated on first run, saved in .env (gitignored) and printed at
startup - without it the server answers 403, so the camera isn't open to
everyone on the Wi-Fi.

Shows the C920 feed with vision_overlay's annotations (hand skeletons, face
boxes + keypoints + owner/unknown, YOLO boxes, status bar). Shares
CameraManager's frames - it never opens the camera itself.

Observer: the robot's own detectors only run when they're needed (faces in
security mode, YOLO on request), so while at least one viewer is connected a
light helper fills the gaps - face detection + recognition and YOLO (320, 2
threads) - only when nothing fresher was published by the real pipeline, and
not at all while a hand is in view: the Pi 4 can't run hands, faces, YOLO and
the wake word at full speed together (measured: hands fell to 3 fps), so the
CPU goes to whatever is being shown. No viewers -> no extra CPU.

Endpoints: /  (page)   /stream.mjpg  (raw MJPEG)   /snapshot.jpg   - all need ?key=

Standalone (camera + overlay + observer only, no assistant):
    python stream_server.py
"""

import http.server
import os
import secrets
import threading
import time
import urllib.parse

import cv2

from vision_overlay import overlay

STREAM_PORT = 8080
DOCK_VIEW = "/dev/shm/slam_view.jpg"   # mono_live's annotated view (dock_camera.py)
STREAM_FPS = 10
JPEG_QUALITY = 70
# Kept modest: with a viewer connected, the stream + observer + gestures once
# starved the wake-word listener of CPU ("Hey Qwen" stopped working). These
# threads also run at lower priority (lower_thread_priority).
OBSERVER_FACE_INTERVAL_S = 0.15      # up to ~6 fps (niced: only uses spare CPU)
OBSERVER_YOLO_INTERVAL_S = 0.5       # up to ~2 fps
ENV_PATH = ".env"


def lower_thread_priority(niceness: int = 10):
    """Runs the calling thread at lower CPU priority (Linux), so the wake-word
    listener always gets the CPU first. No-op elsewhere."""
    import os
    import threading
    try:
        os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), niceness)
    except (AttributeError, OSError):
        pass

PAGE = """<!doctype html><html><head><title>Qwen vision</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>body{margin:0;background:#111;color:#ccc;font:14px system-ui;display:flex;
flex-direction:column;align-items:center}img{max-width:100%;height:auto;margin-top:8px}
p{margin:6px}</style></head><body>
<img src="/stream.mjpg?key=KEY" alt="Qwen camera">
<script>
const cam = document.querySelector("img"), camSrc = cam.src;
document.addEventListener("visibilitychange", () => {
  if (document.hidden) cam.src = "data:,"; else cam.src = camSrc + "&t=" + Date.now();
});
</script>
<p>Qwen - live vision debug view &middot; <a style="color:#8cf" href="/snapshot.jpg?key=KEY">snapshot</a></p>
</body></html>"""


def _stream_key() -> str:
    """QWEN_STREAM_KEY from .env, generated (and saved) on first use."""
    lines = []
    if os.path.exists(ENV_PATH):
        with open(ENV_PATH) as f:
            lines = f.read().splitlines()
        for line in lines:
            if line.startswith("QWEN_STREAM_KEY="):
                return line.split("=", 1)[1].strip()
    key = secrets.token_urlsafe(9)
    with open(ENV_PATH, "w") as f:
        f.write("\n".join(lines + [f"QWEN_STREAM_KEY={key}"]) + "\n")
    os.chmod(ENV_PATH, 0o600)
    return key


class StreamServer:
    def __init__(self, camera, port: int = STREAM_PORT, observe: bool = True, robot=None, dock_camera=None):
        """robot / dock_camera (assistant.py): serve the robot control page at / (the vision
        debug page moves to /vision) and show the tag tracker's view while it has the camera."""
        self.camera = camera
        self.robot = robot
        self.dock_camera = dock_camera
        self.port = port
        self.observe = observe
        self.key = _stream_key()
        self._viewers = 0
        self._viewers_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._server = None
        self._faces = None           # lazy FaceRecognizer for the observer
        self._yolo = None            # lazy 320 detector for the observer

    # ------------------------------------------------------------------ lifecycle

    def start(self):
        server_self = self
        if self.robot is not None:
            import json
            from slam_view import PAGE as CONTROL_PAGE, handle_cmd

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                url = urllib.parse.urlparse(self.path)
                key = urllib.parse.parse_qs(url.query).get("key", [""])[0]
                if not secrets.compare_digest(key, server_self.key):
                    self.send_error(403, "missing or wrong ?key=")
                    return
                if url.path == "/" and server_self.robot is not None:
                    body = CONTROL_PAGE.replace("__KEY__", server_self.key).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif url.path == "/status" and server_self.robot is not None:
                    body = json.dumps(server_self.robot.status()).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    self.wfile.write(body)
                elif url.path in ("/", "/vision"):
                    body = PAGE.replace("KEY", server_self.key).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif url.path == "/snapshot.jpg":
                    jpg = server_self._jpeg()
                    if jpg is None:
                        self.send_error(503, "no camera frame")
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(jpg)))
                    self.end_headers()
                    self.wfile.write(jpg)
                elif url.path == "/stream.mjpg":
                    server_self._serve_mjpeg(self)
                else:
                    self.send_error(404)

            def do_POST(self):
                url = urllib.parse.urlparse(self.path)
                q = urllib.parse.parse_qs(url.query)
                if not secrets.compare_digest(q.get("key", [""])[0], server_self.key):
                    self.send_error(403, "missing or wrong ?key=")
                    return
                if url.path != "/cmd" or server_self.robot is None:
                    self.send_error(404)
                    return
                try:
                    handle_cmd(server_self.robot, q)
                except (ValueError, KeyError) as e:
                    self.send_error(400, str(e))
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"ok")

        self._server = http.server.ThreadingHTTPServer(("0.0.0.0", self.port), Handler)
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        if self.observe:
            threading.Thread(target=self._observer_loop, daemon=True).start()
        print(f"[stream] vision debug view: http://raspberrypi.local:{self.port}/?key={self.key}")

    def stop(self):
        self._stop_event.set()
        if self._server is not None:
            self._server.shutdown()

    # ------------------------------------------------------------------ streaming

    def _jpeg(self):
        if self.dock_camera is not None and self.dock_camera.active:   # tag tracker has the camera
            try:
                with open(DOCK_VIEW, "rb") as f:
                    return f.read()
            except OSError:
                return None
        frame = overlay.annotate(self.camera.get_frame())
        if frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        return buf.tobytes() if ok else None

    def _serve_mjpeg(self, handler):
        handler.send_response(200)
        handler.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        handler.send_header("Cache-Control", "no-cache")
        handler.end_headers()
        lower_thread_priority()
        with self._viewers_lock:
            self._viewers += 1
        try:
            interval = 1.0 / STREAM_FPS
            while not self._stop_event.is_set():
                t = time.time()
                jpg = self._jpeg()
                if jpg is not None:
                    handler.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                        + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                time.sleep(max(0.0, interval - (time.time() - t)))
        except (BrokenPipeError, ConnectionResetError):
            pass                               # viewer closed the tab
        finally:
            with self._viewers_lock:
                self._viewers -= 1

    # ------------------------------------------------------------------ observer

    def _observer_loop(self):
        lower_thread_priority()
        last_face = last_yolo = 0.0
        while not self._stop_event.wait(0.05):
            if self._viewers == 0:
                continue
            frame = self.camera.get_frame()
            if frame is None:
                continue
            now = time.time()
            if overlay.hand_visible():
                continue                       # a hand is in view: let hand tracking have the CPU
            try:
                if now - last_face >= OBSERVER_FACE_INTERVAL_S and not overlay.fresh("faces", 0.5):
                    last_face = now
                    if self._faces is None:
                        from face_recognizer import FaceRecognizer
                        self._faces = FaceRecognizer()
                    overlay.publish_faces(self._faces.classify(frame))
                if now - last_yolo >= OBSERVER_YOLO_INTERVAL_S and not overlay.fresh("objects", 1.0):
                    last_yolo = now
                    if self._yolo is None:
                        from object_detector import Detector, NcnnBackend
                        self._yolo = Detector(NcnnBackend("models/yolo26n_320_ncnn_model", 320, threads=2),
                                              conf=0.4)
                    overlay.publish_objects(self._yolo.detect(frame), source="observer")
            except Exception as e:
                print(f"[stream] observer error: {e}")
                time.sleep(1)


if __name__ == "__main__":
    from camera_manager import CameraManager

    cam = CameraManager()
    cam.start()
    server = StreamServer(cam)
    server.start()
    print("Standalone stream (no hand tracking - that runs inside assistant.py). Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
        cam.stop()
