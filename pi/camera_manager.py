"""
Project Qwen - shared camera manager.

A single background thread owns the C920 (or whatever camera_index points
to), continuously grabbing frames. Both gesture recognition and voice
commands go through this same instance so they don't fight over the device,
and "is recording" is a single source of truth regardless of which path
(voice or gesture) started/stopped it.
"""

import os
import shutil
import subprocess
import threading
import time
import wave

import cv2
import numpy as np
import sounddevice as sd

PHOTOS_DIR = "captures/photos"
VIDEOS_DIR = "captures/videos"
CAMERA_FPS = 20

# Recorded audio track (mic captured alongside video, muxed in via ffmpeg
# when recording stops - cv2.VideoWriter itself is video-only).
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None


class CameraManager:
    def __init__(self, camera_index: int = 0, fps: int = CAMERA_FPS):
        self.camera_index = camera_index
        self.fps = fps
        os.makedirs(PHOTOS_DIR, exist_ok=True)
        os.makedirs(VIDEOS_DIR, exist_ok=True)

        self._cap = None
        self._lock = threading.Lock()
        self._latest_frame = None
        self._running = False
        self._thread = None

        self._writer = None
        self._recording_path = None
        self._silent_path = None
        self._record_start_wall = None
        self._frame_count = 0
        self._capture_fps_hint = float(self.fps)

        self._audio_stream = None
        self._audio_frames = []

        # Optional external mic to ride along on for recording audio,
        # instead of opening a second sd.InputStream of our own - macOS's
        # CoreAudio backend doesn't support two concurrent InputStreams on
        # the same device (confirmed: raises PortAudioError -9986 when
        # assistant.py's voice pipeline mic is also open). Set externally by
        # assistant.py's VoicePipeline (mirrors how it manages tts.mic);
        # stays None for standalone use (gesture_recognizer.py/
        # face_recognizer.py's own __main__ blocks), where the fallback
        # self-contained InputStream below is safe since nothing else has
        # the device open.
        self.mic = None
        self._mic_subscription = None
        self._subscribed_mic = None

        # Optional frame -> annotated-frame function (vision_overlay.annotate),
        # applied to recordings started with annotated=True (security footage);
        # normal recordings and photos stay clean.
        self.annotate = None
        self._record_annotated = False

    def start(self):
        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera index {self.camera_index}")
        # Best-effort guess at the camera's real delivery rate, used only as
        # the VideoWriter's nominal fps and as a fallback if ffmpeg isn't
        # available to retime afterward. Not trusted for anything else -
        # some drivers misreport it.
        reported_fps = self._cap.get(cv2.CAP_PROP_FPS)
        if reported_fps and reported_fps > 1:
            self._capture_fps_hint = reported_fps
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        # Write every frame as it arrives rather than trying to throttle to
        # a target fps here. A wall-clock "write only every N ms" gate was
        # tried and failed: the camera's real delivery rate (~30fps) isn't
        # evenly divisible against an arbitrary target (20fps), so the gate
        # could only fire every 2nd frame at best, capping actual output at
        # 15fps while the file still claimed 20fps -> sped-up playback, same
        # symptom as the original bug just less severe. Instead, every frame
        # gets written and stop_recording() measures the real elapsed time
        # vs. frame count afterward, then ffmpeg retimes the container to
        # match - correct regardless of the camera's true native rate on
        # whatever hardware this runs on (this dev Mac vs. the target Pi 4).
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            with self._lock:
                self._latest_frame = frame
                recording = self._writer is not None
                annotate = recording and self._record_annotated and self.annotate is not None
            if not recording:
                continue
            # Draw the overlay OUTSIDE the lock: the overlay's status bar calls
            # back into this camera (is_recording()), and doing that while
            # holding the (non-reentrant) lock deadlocked the camera the moment
            # an annotated security recording started.
            out = frame
            if annotate:
                try:
                    out = self.annotate(frame)
                except Exception:
                    out = frame
            with self._lock:
                if self._writer is not None:      # may have stopped meanwhile
                    if self._record_start_wall is None:
                        self._record_start_wall = time.time()
                    self._writer.write(out)
                    self._frame_count += 1

    def get_frame(self):
        """Returns the most recent frame (a copy), or None if nothing captured yet."""
        with self._lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    def is_recording(self) -> bool:
        with self._lock:
            return self._writer is not None

    def _audio_callback(self, indata, frames, time_info, status):
        self._audio_frames.append(indata.copy())

    def start_recording(self, annotated: bool = False) -> str | None:
        """Returns the output path, or None if already recording. annotated=True
        burns the vision overlay (boxes, keypoints, status) into the video."""
        with self._lock:
            if self._writer is not None:
                return None
            self._record_annotated = annotated
            if self._latest_frame is None:
                return None  # camera not producing frames yet
            h, w = self._latest_frame.shape[:2]
            filename = f"video_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
            final_path = os.path.join(VIDEOS_DIR, filename)
            silent_path = final_path + ".silent.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(silent_path, fourcc, self._capture_fps_hint, (w, h))
            self._recording_path = final_path
            self._silent_path = silent_path
            self._record_start_wall = None
            self._frame_count = 0

        # Audio capture is best-effort: if it fails for any reason (e.g. a
        # PortAudioError - this exact class of failure has actually
        # happened, when a second InputStream briefly conflicted with an
        # already-open one), fall back to a silent recording rather than
        # raising here. self._writer is already assigned above at this
        # point, so letting an exception escape would leave is_recording()
        # permanently True with no way to stop it short of restarting the
        # process - a raised exception here must not do that.
        try:
            if self.mic is not None:
                # Ride along on the already-open mic stream rather than
                # opening a second one. Remember which mic we subscribed to
                # (rather than relying on self.mic still pointing at it
                # later) in case self.mic gets reassigned/cleared out from
                # under us mid-recording - e.g. security mode arming while a
                # gesture-started recording happens to still be in flight.
                self._subscribed_mic = self.mic
                self._mic_subscription = self._subscribed_mic.subscribe()
            else:
                self._audio_frames = []
                self._audio_stream = sd.InputStream(
                    samplerate=AUDIO_SAMPLE_RATE, channels=AUDIO_CHANNELS, dtype="int16",
                    # Without an explicit blocksize, sounddevice defaults to very
                    # small buffers -> ~1000 callbacks/sec, each grabbing the GIL to
                    # append to _audio_frames. That contention alone was starving
                    # the video capture thread down to ~15fps. 100ms chunks (10
                    # callbacks/sec) is plenty fine-grained for audio and stops the
                    # video thread from being starved.
                    blocksize=int(AUDIO_SAMPLE_RATE * 0.1),
                    callback=self._audio_callback,
                )
                self._audio_stream.start()
        except Exception as e:
            print(f"[camera] Audio capture failed to start ({e!r}) - recording video only.")
            self._mic_subscription = None
            self._subscribed_mic = None
            self._audio_stream = None
        return final_path

    def stop_recording(self) -> str | None:
        """Returns the path that was just finished, or None if not recording."""
        with self._lock:
            if self._writer is None:
                return None
            self._writer.release()
            self._writer = None
            final_path = self._recording_path
            silent_path = self._silent_path
            self._recording_path = None
            self._silent_path = None
            start_wall = self._record_start_wall
            frame_count = self._frame_count

        audio = None
        if self._mic_subscription is not None:
            self._subscribed_mic.unsubscribe(self._mic_subscription)
            raw = b"".join(self._mic_subscription)
            self._mic_subscription = None
            self._subscribed_mic = None
            if raw:
                audio = np.frombuffer(raw, dtype=np.int16)
        elif self._audio_stream is not None:
            self._audio_stream.stop()
            self._audio_stream.close()
            self._audio_stream = None
            if self._audio_frames:
                audio = np.concatenate(self._audio_frames)
            self._audio_frames = []

        elapsed = time.time() - start_wall if start_wall is not None else 0
        measured_fps = frame_count / elapsed if elapsed > 0 and frame_count > 0 else self._capture_fps_hint

        if FFMPEG_AVAILABLE:
            self._finalize_with_ffmpeg(silent_path, measured_fps, audio, final_path)
        else:
            print("ffmpeg not found - video saved without audio, and its "
                  "playback speed may be slightly off (`brew install ffmpeg` "
                  "to fix both).")
            os.replace(silent_path, final_path)

        return final_path

    def _finalize_with_ffmpeg(self, silent_video_path: str, measured_fps: float,
                               audio: np.ndarray | None, final_path: str):
        """Remuxes the silent video against its true measured fps (cv2 wrote
        every frame at capture time but the container's declared fps was
        only ever a best-effort guess) and, if audio was captured, muxes it
        in - cv2.VideoWriter has no audio support of its own. `-c:v copy`
        with a corrected `-r` on the input just rewrites frame timestamps,
        no re-encode needed."""
        wav_path = None
        cmd = ["ffmpeg", "-y", "-r", f"{measured_fps:.3f}", "-i", silent_video_path]

        has_audio = audio is not None and len(audio) > 0
        if has_audio:
            wav_path = silent_video_path + ".wav"
            with wave.open(wav_path, "wb") as wf:
                wf.setnchannels(AUDIO_CHANNELS)
                wf.setsampwidth(2)  # int16
                wf.setframerate(AUDIO_SAMPLE_RATE)
                wf.writeframes(audio.tobytes())
            cmd += ["-i", wav_path, "-c:v", "copy", "-c:a", "aac", "-shortest", final_path]
        else:
            cmd += ["-c:v", "copy", final_path]

        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            print("ffmpeg finalize failed, saving raw video instead:",
                  result.stderr.decode(errors="ignore")[-300:])
            os.replace(silent_video_path, final_path)
        else:
            os.remove(silent_video_path)
        if wav_path is not None and os.path.exists(wav_path):
            os.remove(wav_path)

    def take_photo(self) -> str | None:
        frame = self.get_frame()
        if frame is None:
            return None
        filename = f"photo_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
        path = os.path.join(PHOTOS_DIR, filename)
        cv2.imwrite(path, frame)
        return path

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.stop_recording()
        if self._cap is not None:
            self._cap.release()
