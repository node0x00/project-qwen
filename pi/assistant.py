"""
Project Qwen - AI voice assistant, optimized for Raspberry Pi 4 (4GB).

Pipeline:
  openWakeWord (always-on, cheap)  ->  VAD-clipped command recording
    -> faster-whisper (lazy-loaded, stays resident once used)
    -> intent match against known commands (recording/photo) -> vision action
       (falls through to the LLM if nothing matches)
    -> Ollama qwen2.5:1.5b (short keep_alive so it unloads when idle)
    -> Piper TTS (lazy-loaded, stays resident once used)

Gesture watching (gesture_recognizer.py) runs concurrently on its own thread
against the same CameraManager/VisionActions as the voice path, so a
thumbs-up/OK-sign/peace-sign records/photographs exactly like the equivalent
voice command and saves to the same captures/ folder.

This is also the single orchestrating entry point for security mode
(security_mode.py): saying "Hey Qwen, security mode" arms it, shutting down
voice/camera/gestures except a PIR motion listener (real gpiozero on a Pi,
a keypress mock elsewhere - see pir_sensor.py), then brings the camera back
for continuous face recognition once motion is detected. See
security_mode.py's module docstring for the full state machine.

Design goals given hardware constraints (4GB RAM, battery-powered, shared
with vision/navigation subsystems):
  - Only the wake-word classifier runs continuously. Everything heavier is
    loaded on first use and the biggest consumer (the LLM) is configured to
    unload itself from Ollama after a short idle period.
  - Mic input is muted while the robot is speaking, to avoid the speaker's
    own output leaking into the mic and causing false wake triggers or
    garbage transcriptions.
  - Security mode's ARMED_IDLE state fully tears down the mic/wake-word
    listener (not just pauses it) to actually free the device/CPU, since
    that's the whole point of the low-power idle state.

Run:
    python assistant.py

Requires:
    ollama serve                 (with qwen2.5:1.5b already pulled)
    models/hey_qwen.onnx         (from the openWakeWord Colab notebook)
"""

import collections
import gc
import os
import re
import subprocess
import threading
import time

import numpy as np
import sounddevice as sd

import audio_out
import webrtcvad
from openwakeword.model import Model as OWWModel
import ollama

from camera_manager import CameraManager
from vision_actions import VisionActions
from intent_router import match_intent, on_off_command, MOVEMENT_INTENTS
from flashlight import get_flashlight
from gesture_recognizer import GestureController
from security_mode import SecurityController, SecurityState
from esp32_link import ESP32Link
from movement_actions import MovementActions
from idle_personality import IdlePersonality
from status_led import StatusLED
from scene_actions import SceneActions
from smart_home import SmartHomeHub
from telegram_notify import TelegramNotifier
from stream_server import StreamServer
from dock_camera import DockCamera
from power_manager import PowerManager
from robot_control import RobotController
from lost_tag_search import LostTagSearch
from person_tracker import PersonTracker
from sleep_mode import SleepMode
from pir_sensor import get_pir_sensor
from vision_overlay import overlay

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

SAMPLE_RATE = 16000

# openWakeWord expects 80ms frames (1280 samples @ 16kHz)
OWW_FRAME_MS = 80
OWW_FRAME_SAMPLES = int(SAMPLE_RATE * OWW_FRAME_MS / 1000)
WAKE_MODEL_PATH = "models/hey_qwen.onnx"
WAKE_THRESHOLD = 0.25            # lower = more sensitive (more false wakes); was 0.5, 0.35, 0.30
# Automatic gain for quiet speech (the C920 mic is already at its +50 dB hardware maximum).
# The level follows the loudest recent sound: fast up when someone speaks, slow back down,
# so in a quiet room the gain creeps up and soft speech reaches the wake-word model and the
# speech detector at a normal level. Video recordings keep the raw audio.
AGC_TARGET_RMS = 2500            # int16 RMS that speech is lifted toward
AGC_MAX_GAIN = 4.0               # +12 dB at most, so silence isn't pumped into loud noise
AGC_RELEASE = 0.995              # per read: how slowly the level estimate falls (~10 s)

WAKE_NEAR_MISS = 0.15            # scores above this (but below threshold) get logged, for tuning

# VAD-based command recording (after wake word fires)
VAD_FRAME_MS = 30
VAD_FRAME_SAMPLES = int(SAMPLE_RATE * VAD_FRAME_MS / 1000)
VAD_AGGRESSIVENESS = 3          # 0-3; 3 = strictest about what counts as speech, so
                                 # steady background noise doesn't keep the recording open
PADDING_MS = 300
PADDING_FRAMES = PADDING_MS // VAD_FRAME_MS
MAX_UTTERANCE_S = 8
# QWEN_DEBUG_AUDIO=1 saves every command recording to /tmp/qwen_rec_*.wav and logs
# the VAD's per-frame decisions - for diagnosing end-of-speech detection.
DEBUG_AUDIO = os.environ.get("QWEN_DEBUG_AUDIO") == "1"
FOLLOWUP_TIMEOUT_S = 6

WHISPER_MODEL_SIZE = "tiny.en"   # smallest viable on Pi4 CPU; bump to base.en if accuracy is lacking
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"

OLLAMA_MODEL = "qwen2.5:1.5b"
# Kept loaded permanently (-1) and warmed up at startup: a cold load costs ~30 s
# on the Pi 4, which made the first answer after a quiet spell take a minute.
# Costs ~1.2 GB of RAM all the time - set to e.g. "15m" to unload when idle
# again if vision/navigation need the memory.
OLLAMA_KEEP_ALIVE = -1
OLLAMA_NUM_PREDICT = 50          # hard cap on reply length - bounds worst-case generation time
SYSTEM_PROMPT = (
    "You are Qwen, a helpful voice assistant running on a small home robot. "
    "Answer in one or two short sentences - your replies are read aloud."
)

PIPER_MODEL_PATH = "models/en_US-lessac-medium.onnx"

# Idle lift animation only runs once there's been no wake word / command for this long.
CONVERSATION_QUIET_S = 60

# Whisper decoding hints - biases tiny.en toward the command vocabulary so
# "come forward" doesn't come out as "blue forward".
WHISPER_HOTWORDS = ("Qwen look up look down turn left turn right come forward back up "
                    "take a photo start recording stop recording flashlight lights security mode")

MIC_MUTE_TAIL_S = 0.3            # keep ignoring mic input briefly after TTS finishes (echo settle)


# --------------------------------------------------------------------------
# Continuous mic stream, shared by the wake-word listener and the VAD recorder
# --------------------------------------------------------------------------

class MicStream:
    """Wraps a single InputStream so wake-word listening and command
    recording share one open audio device instead of fighting over it.

    Also fans out a copy of every frame read to any subscribers (see
    subscribe()/unsubscribe()) - this is how CameraManager captures audio
    for video recordings without opening a second InputStream of its own.
    macOS's CoreAudio backend doesn't support two concurrent InputStreams on
    the same device (confirmed: opening a second one while this one is open
    raises PortAudioError -9986), so recording audio must ride along on this
    same stream rather than each having their own."""

    def __init__(self):
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="int16"
        )
        self._stream.start()
        self.muted_until = 0.0
        self.overflows = 0
        self._last_overflow_log = 0.0
        self._subscribers: list[collections.deque] = []
        self._sub_lock = threading.Lock()
        self._agc_level = float(AGC_TARGET_RMS)
        self.gain = 1.0

    def read(self, n_samples: int) -> bytes:
        data, overflowed = self._stream.read(n_samples)
        if overflowed:
            # The listener fell behind the mic (CPU starved) - audio was lost.
            self.overflows += 1
            now = time.time()
            if now - self._last_overflow_log > 10:
                self._last_overflow_log = now
                print(f"[mic] audio dropped - listener falling behind ({self.overflows} so far)")
        raw = (b"\x00" * (n_samples * 2)) if time.time() < self.muted_until else data.tobytes()
        with self._sub_lock:
            for buf in self._subscribers:
                buf.append(raw)
        return self._agc(raw)

    def _agc(self, raw: bytes) -> bytes:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        if not samples.size:
            return raw
        rms = float(np.sqrt(np.mean(samples * samples)))
        if rms > self._agc_level:
            self._agc_level = 0.5 * self._agc_level + 0.5 * rms        # fast attack
        else:
            self._agc_level = max(1.0, AGC_RELEASE * self._agc_level + (1 - AGC_RELEASE) * rms)
        self.gain = min(AGC_MAX_GAIN, max(1.0, AGC_TARGET_RMS / self._agc_level))
        if self.gain <= 1.01:
            return raw
        return np.clip(samples * self.gain, -32768, 32767).astype(np.int16).tobytes()

    def subscribe(self) -> collections.deque:
        """Returns a deque that will receive a copy of every frame read from
        here on, until unsubscribe() is called. Note: only fills while
        something is actively calling read() (i.e. the voice loop is
        running), which is always true whenever this MicStream exists."""
        buf: collections.deque = collections.deque()
        with self._sub_lock:
            self._subscribers.append(buf)
        return buf

    def unsubscribe(self, buf: collections.deque):
        with self._sub_lock:
            if buf in self._subscribers:
                self._subscribers.remove(buf)

    def mute_for(self, seconds: float):
        self.muted_until = time.time() + seconds

    def close(self):
        self._stream.stop()
        self._stream.close()


# --------------------------------------------------------------------------
# Wake word (always-on, low power)
# --------------------------------------------------------------------------

class WakeWordListener:
    def __init__(self, mic: MicStream):
        self.mic = mic
        # Pinned to openwakeword==0.4.0 (see requirements.txt) - versions
        # >=0.5.0 require tflite-runtime on Linux even though we only use
        # the ONNX backend, and tflite-runtime has no wheels for Python
        # >=3.12, which blocks installing on the Raspberry Pi entirely.
        # 0.4.0's constructor param is `wakeword_model_paths`, not
        # `wakeword_models`, and it has no `inference_framework` param
        # (only ever supported ONNX). It also bundles its melspectrogram/
        # embedding feature models directly in the package instead of
        # lazily downloading them, so no download_models() call is needed.
        self.model = OWWModel(wakeword_model_paths=[WAKE_MODEL_PATH])
        self.wake_key = list(self.model.models.keys())[0]

    def wait_for_wake(self, stop_event: threading.Event | None = None) -> bool:
        """Blocks until the wake word fires, or `stop_event` is set (checked
        once per ~80ms frame) - lets security mode stop the voice loop
        cleanly instead of fighting this blocking call. Returns True if the
        wake word fired, False if stopped early."""
        while stop_event is None or not stop_event.is_set():
            frame_bytes = self.mic.read(OWW_FRAME_SAMPLES)
            frame = np.frombuffer(frame_bytes, dtype=np.int16)
            prediction = self.model.predict(frame)
            score = prediction[self.wake_key]
            if score > WAKE_THRESHOLD:
                print(f"[wake] score {score:.2f}")
                self.model.reset()
                return True
            if score > WAKE_NEAR_MISS:
                print(f"[wake] near miss {score:.2f} (threshold {WAKE_THRESHOLD})")
        return False


# --------------------------------------------------------------------------
# VAD-triggered command recording (after wake word)
# --------------------------------------------------------------------------

class CommandRecorder:
    def __init__(self, mic: MicStream):
        self.mic = mic
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self._debug_count = 0

    def record_utterance(self, timeout_s: float | None = None,
                          stop_event: threading.Event | None = None) -> np.ndarray | None:
        ring_buffer = collections.deque(maxlen=PADDING_FRAMES)
        triggered = False
        voiced_frames = []
        decisions = []   # per-frame VAD result after triggering (debug only)
        start_time = time.time()

        while stop_event is None or not stop_event.is_set():
            frame = self.mic.read(VAD_FRAME_SAMPLES)
            is_speech = self.vad.is_speech(frame, SAMPLE_RATE)

            if not triggered:
                ring_buffer.append((frame, is_speech))
                num_voiced = sum(1 for _, s in ring_buffer if s)
                if num_voiced > 0.5 * ring_buffer.maxlen:
                    triggered = True
                    voiced_frames.extend(f for f, _ in ring_buffer)
                    ring_buffer.clear()
                elif timeout_s is not None and (time.time() - start_time) > timeout_s:
                    return None
            else:
                voiced_frames.append(frame)
                if DEBUG_AUDIO:
                    decisions.append(is_speech)
                ring_buffer.append((frame, is_speech))
                num_unvoiced = sum(1 for _, s in ring_buffer if not s)
                elapsed = len(voiced_frames) * VAD_FRAME_MS / 1000
                if num_unvoiced > 0.9 * ring_buffer.maxlen or elapsed > MAX_UTTERANCE_S:
                    break
        else:
            return None  # loop condition failed (stop_event set) before any speech triggered

        raw = b"".join(voiced_frames)
        if DEBUG_AUDIO:
            self._save_debug(raw, decisions)
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        return audio

    def _save_debug(self, raw: bytes, decisions: list):
        import wave
        self._debug_count += 1
        path = f"/tmp/qwen_rec_{self._debug_count}.wav"
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(raw)
        # One char per 30 ms frame: '#' = speech, '.' = not speech.
        pattern = "".join("#" if d else "." for d in decisions)
        print(f"[debug] saved {path}; VAD after trigger: {pattern}")


# --------------------------------------------------------------------------
# Speech-to-text (lazy-loaded, stays resident once used)
# --------------------------------------------------------------------------

class Transcriber:
    def __init__(self):
        self._model = None

    def _ensure_loaded(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            print(f"Loading faster-whisper ({WHISPER_MODEL_SIZE})...")
            self._model = WhisperModel(
                WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE
            )

    def preload(self):
        self._ensure_loaded()

    def transcribe(self, audio: np.ndarray) -> str:
        self._ensure_loaded()
        segments, _ = self._model.transcribe(audio, language="en", beam_size=1,
                                             hotwords=WHISPER_HOTWORDS,
                                             without_timestamps=True)
        return " ".join(seg.text.strip() for seg in segments).strip()


# --------------------------------------------------------------------------
# LLM (Ollama, with a short keep_alive so it frees RAM when idle)
# --------------------------------------------------------------------------

_SENTENCE_END = re.compile(r"[.!?]+[\"')\]]*\s")


class LLMClient:
    def __init__(self, model: str = OLLAMA_MODEL):
        self.model = model
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]

    def warm_up(self):
        """Loads the model into RAM and caches the system prompt, so the first
        real question doesn't pay the ~30 s cold load."""
        ollama.chat(model=self.model, messages=[self.history[0], {"role": "user", "content": "hi"}],
                    keep_alive=OLLAMA_KEEP_ALIVE, options={"num_predict": 1})

    def ask_stream(self, text: str):
        """Yields the reply one sentence at a time as it's generated, so the
        first sentence can be spoken while the rest is still being written."""
        self.history.append({"role": "user", "content": text})
        stream = ollama.chat(
            model=self.model,
            messages=self.history,
            keep_alive=OLLAMA_KEEP_ALIVE,
            options={"num_predict": OLLAMA_NUM_PREDICT},
            stream=True,
        )
        reply, pending = "", ""
        for chunk in stream:
            piece = chunk["message"]["content"]
            reply += piece
            pending += piece
            # Sentence end = . ! ? followed by whitespace ("3.5" isn't a break).
            while (m := _SENTENCE_END.search(pending)) is not None:
                sentence, pending = pending[:m.end()].strip(), pending[m.end():]
                if sentence:
                    yield sentence
        if pending.strip():
            yield pending.strip()
        self.history.append({"role": "assistant", "content": reply})
        if len(self.history) > 12:
            self.history = [self.history[0]] + self.history[-10:]


# --------------------------------------------------------------------------
# Text-to-speech (lazy-loaded, stays resident once used)
# --------------------------------------------------------------------------

class TTS:
    def __init__(self, mic: MicStream | None = None):
        # mic is optional/settable rather than fixed at construction time,
        # since security mode tears down and rebuilds the mic across
        # ARMED_IDLE/ARMED_ACTIVE transitions while TTS itself stays a
        # single persistent instance for the program's whole lifetime (so
        # the loaded Piper voice never has to reload on arm/disarm cycles).
        self.mic = mic
        self._voice = None
        # speak() is called from several threads (voice loop, gestures, ESP32
        # link warnings); sounddevice's play()/wait() is global, so overlapping
        # calls would cut each other off. One utterance at a time.
        self._lock = threading.Lock()
        self._cache: dict[str, tuple[np.ndarray, int]] = {}   # pre-synthesized phrases

    def preload(self, phrases=()):
        """Loads the voice and pre-synthesizes short fixed phrases ("Yes?") so
        they play instantly."""
        self._ensure_loaded()
        for text in phrases:
            self._cache[text] = self._synthesize(text)

    def _synthesize(self, text: str):
        # Piper reads "Qwen" as the letter Q + "wen"; spell it the way it is said.
        text = re.sub(r"\bqwen\b", "Kwen", text, flags=re.IGNORECASE)
        audio_chunks = []
        sample_rate = None
        for chunk in self._voice.synthesize(text):
            sample_rate = chunk.sample_rate
            audio_chunks.append(np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16))
        if not audio_chunks:
            return None, None
        return np.concatenate(audio_chunks), sample_rate

    def _ensure_loaded(self):
        if self._voice is None:
            from piper.voice import PiperVoice
            print("Loading Piper voice...")
            self._voice = PiperVoice.load(PIPER_MODEL_PATH)

    def speak(self, text: str):
        if not text.strip():
            return
        with self._lock:
            self._speak_locked(text)

    def _speak_locked(self, text: str):
        self._ensure_loaded()
        print(f"Qwen: {text}")

        audio, sample_rate = self._cache.get(text) or self._synthesize(text)
        if audio is None:
            return
        duration_s = len(audio) / sample_rate

        # Mute mic input for the duration of playback + a short settle tail,
        # so the speaker's own output can't leak back in as a false trigger.
        # Only applies if a mic is currently live (it isn't during
        # security mode's ARMED_IDLE, when the mic is fully torn down).
        if self.mic is not None:
            self.mic.mute_for(duration_s + MIC_MUTE_TAIL_S)
        audio_out.play(audio, sample_rate)   # own stream - safe alongside the alarm


# --------------------------------------------------------------------------
# Voice pipeline - a fully restartable unit (mic + wake word + VAD recorder)
# --------------------------------------------------------------------------

class VoicePipeline:
    """Owns the mic/wake-word/VAD-recorder trio as a single restartable
    unit, so security mode can really tear it down (freeing the audio
    device and unloading the wake-word model) during ARMED_IDLE and rebuild
    it fresh on resume - not just pause a loop, since the actual point of
    ARMED_IDLE is to free the device/CPU. Runs its own daemon thread; stop()
    is cooperative (a threading.Event checked between mic-read frames)
    rather than fighting the blocking mic reads."""

    def __init__(self, tts: TTS, transcriber: Transcriber, llm: LLMClient, handle_command_fn,
                 camera: CameraManager | None = None):
        self.tts = tts
        self.transcriber = transcriber
        self.llm = llm
        self.handle_command = handle_command_fn
        self.camera = camera

        self.mic = None
        self.wake_listener = None
        self.recorder = None
        self._thread = None
        self._stop_event = threading.Event()
        self.last_activity = 0.0   # time of the last wake word / command (idle animation)
        # Called with True from the wake word until the reply is done, False
        # after - main() uses it to pause gesture tracking so STT/LLM get the CPU.
        self.on_busy = lambda busy: None
        # Called with "listening" / "thinking" / "idle" as an interaction
        # progresses - drives the white status LED.
        self.on_phase = lambda phase: None

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self.mic = MicStream()
        self.tts.mic = self.mic
        if self.camera is not None:
            self.camera.mic = self.mic
        self.wake_listener = WakeWordListener(self.mic)
        self.recorder = CommandRecorder(self.mic)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        # If stop() is called from the voice thread itself (e.g. handling
        # "security mode" mid-command), joining self._thread here would be
        # the thread joining itself - RuntimeError at best, a deadlock at
        # worst. In that case just set the event and close the mic; _run()
        # notices the event and returns on its own right after
        # handle_command() returns control to it.
        if self._thread is not None and threading.current_thread() is not self._thread:
            self._thread.join(timeout=5.0)
        if self.mic is not None:
            self.mic.close()
            self.mic = None
        self.tts.mic = None
        if self.camera is not None:
            self.camera.mic = None
        self._thread = None

    def _run(self):
        print('Voice pipeline up - listening for "Hey Qwen"')
        while not self._stop_event.is_set():
            if not self.wake_listener.wait_for_wake(stop_event=self._stop_event):
                break
            print("[wake word detected]")
            self.last_activity = time.time()
            self.on_busy(True)
            self.on_phase("listening")
            try:
                self.tts.speak("Yes?")

                audio = self.recorder.record_utterance(
                    timeout_s=FOLLOWUP_TIMEOUT_S, stop_event=self._stop_event)
                if audio is None:
                    continue

                print(f"[recorded] {len(audio) / SAMPLE_RATE:.1f}s of audio")
                self.on_phase("thinking")
                t0 = time.time()
                transcript = self.transcriber.transcribe(audio)
                if not transcript:
                    continue
                print(f"[heard] {transcript}  ({time.time() - t0:.1f}s to transcribe)")

                self.handle_command(transcript)
                self.last_activity = time.time()
            finally:
                self.on_busy(False)
                self.on_phase("idle")

            gc.collect()  # nudge Python to release Whisper/audio buffers between turns


# --------------------------------------------------------------------------
# Main - builds every subsystem once, wires callbacks, then idles
# --------------------------------------------------------------------------

def main():
    # Print every thread's Python stack if a native library crashes the process
    # (segfault / abort such as "free(): invalid pointer") - otherwise those
    # crashes leave no trace in the log.
    import faulthandler
    faulthandler.enable()

    transcriber = Transcriber()
    llm = LLMClient()
    tts = TTS()

    camera = CameraManager()
    camera.start()
    flashlight = get_flashlight()
    notifier = TelegramNotifier()   # photos/videos + security alerts to the owner's phone
    notifier.start()
    vision_actions = VisionActions(camera, tts, flashlight, notifier=notifier)

    esp32 = ESP32Link(speak=tts.speak)
    esp32.start()
    movement = MovementActions(esp32, tts)
    # Docking / driving (robot_control.py) on the same ESP32 link; while it drives itself the
    # dock-tag tracker borrows the camera (gestures pause), see dock_camera.py.
    gesture_paused_for_dock = {"was_running": False}

    def pause_vision_for_dock():
        gesture_paused_for_dock["was_running"] = (gesture_state["thread"] is not None
                                                  and gesture_state["thread"].is_alive())
        stop_gestures()

    def resume_vision_after_dock():
        if gesture_paused_for_dock["was_running"] and security.state == SecurityState.OFF:
            start_gestures()

    dock_camera = DockCamera(camera, pause_vision=lambda: pause_vision_for_dock(),
                             resume_vision=lambda: resume_vision_after_dock())
    robot = RobotController(link=esp32, camera_hook=dock_camera)
    robot.lost_tag_search = LostTagSearch(robot, dock_camera)   # ORB-SLAM search when a 360 finds no tags
    robot.say = tts.speak
    # Room lights on while docking (the camera sees the tags better); switched back off
    # afterwards only if Qwen knows they were off before.
    dock_lights = {"before": None}

    def dock_lights_on():
        dock_lights["before"] = hub.lights_known
        hub.lights_on()

    def dock_lights_restore(result):
        if dock_lights["before"] is False:
            hub.lights_off()
    robot.on_dock_start = dock_lights_on
    robot.on_dock_end = dock_lights_restore

    # Security mode turns the robot to keep a person in view; if that took it off the
    # charger, it goes back to the dock once the person is gone (or the owner disarmed).
    def redock_after_tracking():
        if robot.mode == "idle":
            print("[track] left the dock to track someone - docking again")
            robot.start_dock_sequence()
    tracker = PersonTracker(esp32, robot, is_charging=lambda: power.charging is True,
                            on_left_dock=redock_after_tracking)
    hub = SmartHomeHub()      # lights/door/keypad; started once security exists
    security = None  # assigned below; closures resolve it at call time
    scene = SceneActions(camera, tts, esp32,
                         allow_movement=lambda: security.state == SecurityState.OFF)

    # Gesture watching runs on its own thread against the same CameraManager
    # and VisionActions the voice path uses, so a thumbs-up/OK-sign/peace-sign
    # saves to the same captures/ folder and gets the same spoken confirmation
    # as the equivalent voice command. Wrapped in start/stop closures so
    # security mode can turn it off during ARMED_ACTIVE (a trespasser's own
    # gesture shouldn't be able to stop their own recording).
    gesture_controller = GestureController(
        camera_manager=camera,
        on_start_recording=vision_actions.start_recording,
        on_stop_recording=vision_actions.stop_recording,
        on_take_photo=vision_actions.take_photo,
        is_recording_fn=camera.is_recording,
        on_toggle_flashlight=vision_actions.toggle_flashlight,
        on_lock_door=lambda: (hub.door_lock(), tts.speak("Locking the door.")),
    )
    gesture_state = {"thread": None, "stop_event": None}

    def start_gestures():
        if gesture_state["thread"] is not None and gesture_state["thread"].is_alive():
            return
        stop_event = threading.Event()
        thread = threading.Thread(
            target=gesture_controller.run_forever,
            kwargs={"stop_event": stop_event}, daemon=True,
        )
        gesture_state["thread"] = thread
        gesture_state["stop_event"] = stop_event
        thread.start()

    def stop_gestures():
        thread, stop_event = gesture_state["thread"], gesture_state["stop_event"]
        if thread is None:
            return
        stop_event.set()
        thread.join(timeout=2.0)
        gesture_state["thread"] = None
        gesture_state["stop_event"] = None


    def handle_command(transcript: str):
        """Routes a transcript to a vision action if it matches a known
        command, otherwise falls through to the LLM."""
        intent = match_intent(transcript)
        if intent == "security_mode":
            security.arm(source="voice")
            return
        if intent == "alarm":
            security.voice_alarm()
            return
        # Security mode has sole authority over the camera's recording while
        # ARMED_ACTIVE - neither the owner nor a trespasser can voice-stop
        # the autonomous security recording. Same for the flashlight and
        # movement, so a trespasser can't switch the light or turn the camera away.
        if (intent in ("start_recording", "stop_recording", "take_photo", "flashlight", "room_lights")
                or intent in MOVEMENT_INTENTS) and security.is_armed_active():
            return
        if intent in MOVEMENT_INTENTS and robot.mode == "auto":
            tts.speak("I'm busy driving right now.")
            return
        if intent == "start_recording":
            vision_actions.start_recording()
        elif intent == "stop_recording":
            vision_actions.stop_recording()
        elif intent == "take_photo":
            vision_actions.take_photo()
        elif intent == "what_do_you_see":
            scene.describe_scene()
        elif intent == "anyone_here":
            scene.anyone_here()
        elif intent == "look_up":
            movement.look_up()
        elif intent == "look_down":
            movement.look_down()
        elif intent == "turn_left":
            movement.turn_left()
        elif intent == "turn_right":
            movement.turn_right()
        elif intent == "move_forward":
            movement.forward()
        elif intent == "move_backward":
            movement.back_up()
        elif intent == "room_lights":
            mode = on_off_command(transcript)
            if mode == "toggle":
                mode = "off" if hub.lights_known else "on"
            hub.lights_on() if mode == "on" else hub.lights_off()
            tts.speak("Lights on." if mode == "on" else "Lights off.")
        elif intent == "flashlight":
            mode = on_off_command(transcript)
            if mode == "on":
                vision_actions.flashlight_on()
            elif mode == "off":
                vision_actions.flashlight_off()
            else:
                vision_actions.toggle_flashlight()
        else:
            t0 = time.time()
            for i, sentence in enumerate(llm.ask_stream(transcript)):
                if i == 0:
                    print(f"[llm] first sentence after {time.time() - t0:.1f}s")
                tts.speak(sentence)

    voice_pipeline = VoicePipeline(tts, transcriber, llm, handle_command, camera=camera)
    security = SecurityController(
        camera=camera, tts=tts,
        start_voice=voice_pipeline.start, stop_voice=voice_pipeline.stop,
        start_gestures=start_gestures, stop_gestures=stop_gestures,
        start_camera=dock_camera.start_camera, stop_camera=dock_camera.stop_camera,
        hub=hub, link=esp32, notifier=notifier, flashlight=flashlight, tracker=tracker,
    )
    hub.on_event = security.on_hub_event
    hub.start()

    # Computer-vision debug overlay: live stream + annotated security footage/photos.
    camera.annotate = overlay.annotate

    def vision_status():
        parts = [f"security: {security.state.value}"]
        if camera.is_recording():
            parts.append("REC")
        if security.alarm.is_playing():
            parts.append("ALARM")
        if esp32.state.lift_pos is not None:
            parts.append(f"lift {esp32.state.lift_pos}")
        blocked = [k for k, v in esp32.state.ir.items() if v]
        if blocked:
            parts.append("IR " + ",".join(blocked))
        return parts

    overlay.status_fn = vision_status
    stream = None
    if os.environ.get("QWEN_STREAM", "1") != "0":
        stream = StreamServer(camera, robot=robot, dock_camera=dock_camera)
        stream.start()

    idle = IdlePersonality(esp32, can_animate=lambda: (
        security.state == SecurityState.OFF
        and robot.mode == "idle"
        and not sleep.asleep
        and power.charging is False
        and not camera.is_recording()
        and time.time() - voice_pipeline.last_activity > CONVERSATION_QUIET_S))

    voice_pipeline.on_busy = gesture_controller.set_paused

    status_led = StatusLED(esp32, is_recording=camera.is_recording,
                           is_armed=lambda: security.is_armed())
    voice_pipeline.on_phase = status_led.set_voice_phase
    status_led.start()

    def warm_up():
        t0 = time.time()
        tts.preload(phrases=("Yes?",))
        transcriber.preload()
        print(f"[warmup] voice + speech-to-text ready ({time.time() - t0:.1f}s)")
        try:
            llm.warm_up()
            print(f"[warmup] LLM loaded and kept in RAM ({time.time() - t0:.1f}s)")
        except Exception as e:
            print(f"[warmup] LLM warm-up failed ({e}) - first question will be slow")

    threading.Thread(target=warm_up, daemon=True).start()

    power = PowerManager(esp32, robot, speak=tts.speak, notify=notifier.notify)

    # Software emergency stop from Telegram: "stop" halts the wheels and the lift and blocks
    # every motion command; "resume" lifts the block and restarts docking if it was docking.
    estop = {"task": None}

    def telegram_command(text):
        word = text.strip().lower().strip("!. ")
        if word == "stop":
            estop["task"] = robot.task if robot.mode == "auto" else None
            robot.stop("STOP from Telegram")
            esp32.halt()
            print("[estop] STOP from Telegram", flush=True)
            notifier.notify("🛑 Stopped. Wheels and lift are locked - send \"resume\" to continue.")
        elif word == "resume":
            if not esp32.halted:
                notifier.notify("Already running.")
                return
            esp32.resume()
            print("[estop] resumed from Telegram", flush=True)
            if estop["task"] == "_dock_sequence" and not power.charging:
                robot.start_dock_sequence()
                notifier.notify("▶️ Resumed - docking again.")
            else:
                notifier.notify("▶️ Resumed.")
            estop["task"] = None
    notifier.on_command = telegram_command

    # Control-page buttons: arm security mode, alarm on/off, speaker volume.
    vol_cache = {"t": 0.0, "v": None}

    def speaker_volume():
        if time.time() - vol_cache["t"] > 2:
            try:
                out = subprocess.run(["wpctl", "get-volume", "@DEFAULT_AUDIO_SINK@"],
                                     capture_output=True, text=True, timeout=2).stdout
                vol_cache["v"] = round(float(out.split()[1]) * 100)
            except Exception:
                pass
            vol_cache["t"] = time.time()
        return vol_cache["v"]

    def set_volume(pct):
        pct = max(0, min(100, int(float(pct))))
        subprocess.run(["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", f"{pct / 100:.2f}"], timeout=3)
        vol_cache.update(t=time.time(), v=pct)

    def page_arm(_):
        if security.state == SecurityState.OFF:
            security.arm(source="control page")

    robot.page_hooks = {"arm": page_arm, "alarm": lambda _: security.toggle_manual_alarm(),
                        "volume": set_volume,
                        "demo_full": lambda _: threading.Thread(target=power.demo_full, daemon=True).start()}
    robot.extra_status = lambda: {"security": security.state.value, "alarm": security.alarm.is_playing(),
                                  "volume": speaker_volume()}
    # The lift servo's current draw upsets the charging contacts: no lift moves while charging
    # (owner, 2026-10-17) - idle animations, "look up/down". Security mode may still use it (the
    # remote-unlock face check needs the lift up, owner's call), and it may always go back down
    # to 0. Until the first battery reading (charging unknown) it counts as charging.
    esp32.lift_allowed = lambda deg: (power.charging is False or deg <= 0
                                      or security.state != SecurityState.OFF
                                      or security.remote_lift_pending)
    robot.start()
    power.start()

    # Idle mode: no motion for 2 minutes -> camera, gestures and voice off; PIR wakes it.
    def go_to_sleep():
        voice_pipeline.stop()
        stop_gestures()
        dock_camera.stop_camera()
        # tell security the camera is off, so a remote unlock / motion arriving while asleep
        # switches it back on (2026-10-17: remote unlock while asleep -> no camera, no face check)
        security._camera_on = False

    def wake_up():
        if security.state != SecurityState.OFF:
            return                              # security mode runs the camera/voice itself
        if dock_camera.active:                  # docking: it hands these back when it's done
            dock_camera.camera_wanted = True
            gesture_paused_for_dock["was_running"] = True
        else:
            dock_camera.start_camera()
            start_gestures()
        security._camera_on = True
        voice_pipeline.start()

    sleep = SleepMode(get_pir_sensor(), go_to_sleep, wake_up, busy_fns=[
        lambda: robot.mode != "idle",
        camera.is_recording,
        lambda: security.state != SecurityState.OFF,
        lambda: time.time() - voice_pipeline.last_activity < 30,
        lambda: stream is not None and stream._viewers > 0,
    ])

    start_gestures()
    voice_pipeline.start()
    idle.start()
    sleep.start()
    print('Idle - listening for "Hey Qwen", watching gestures, say '
          '"security mode" to arm (Ctrl+C to quit)\n')

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        sleep.stop()
        power.stop()
        robot.close()
        dock_camera.shutdown()
        idle.stop()
        status_led.stop()
        voice_pipeline.stop()
        stop_gestures()
        security.shutdown()
        hub.stop()
        notifier.stop()
        if stream is not None:
            stream.stop()
        flashlight.off()
        esp32.close()
        camera.stop()


if __name__ == "__main__":
    main()
