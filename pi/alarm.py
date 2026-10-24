"""
Project Qwen - looping alarm playback for security mode.

Plays a user-supplied WAV file on loop until stopped. This project doesn't
ship an alarm sound - drop a royalty-free siren/alarm clip (e.g. from
freesound.org) at sounds/alarm.wav to enable it. Missing file -> a warning
is printed and start() is a no-op, not a crash.

Uses sounddevice (already a dependency elsewhere in this project) +
soundfile for decoding, rather than adding a new heavier audio library.
"""

import os
import threading

import audio_out

ALARM_SOUND_PATH = "sounds/alarm.wav"


class Alarm:
    def __init__(self, sound_path: str = ALARM_SOUND_PATH):
        self.sound_path = sound_path
        self._data = None
        self._samplerate = None
        self._thread = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> bool:
        if self._data is not None:
            return True
        if not os.path.exists(self.sound_path):
            print(f"[alarm] {self.sound_path} not found - supply an alarm WAV "
                  f"there to enable the security-mode siren (see README).")
            return False
        import soundfile as sf
        self._data, self._samplerate = sf.read(self.sound_path, dtype="float32")
        return True

    def is_playing(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def start(self):
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            if not self._ensure_loaded():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def _loop(self):
        # Own output stream per loop (audio_out) - never the shared sd.play()
        # stream, which the TTS voice also uses from another thread.
        while not self._stop_event.is_set():
            audio_out.play(self._data, self._samplerate, stop_event=self._stop_event)

    def stop(self):
        with self._lock:
            self._stop_event.set()
            thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=1.0)


if __name__ == "__main__":
    import time

    print(f"Testing alarm playback from {ALARM_SOUND_PATH} for 5 seconds (Ctrl+C to stop early)...")
    alarm = Alarm()
    alarm.start()
    try:
        time.sleep(5)
    except KeyboardInterrupt:
        pass
    finally:
        alarm.stop()
        print("Stopped.")
