"""
Project Qwen - thread-safe audio playback.

sounddevice.play()/wait()/stop() share ONE module-global stream. The TTS
voice and the security alarm used it from different threads at the same time
(e.g. "Welcome back" starting while the alarm stopped), which can free a
stream another thread is still using - the assistant crashed with
"free(): invalid pointer" right after a disarm. Here every playback opens its
own OutputStream, so voice and alarm simply mix instead of fighting.
"""

import threading

import numpy as np
import sounddevice as sd

BLOCK_FRAMES = 2048          # ~46 ms at 44.1 kHz - how often stop requests are checked


def play(audio: np.ndarray, samplerate: int, stop_event: threading.Event | None = None) -> bool:
    """Plays `audio` (int16 or float32; mono 1-D or (frames, channels)) on its own
    stream, blocking until done. Returns False if stopped early via stop_event."""
    data = audio if audio.ndim == 2 else audio.reshape(-1, 1)
    with sd.OutputStream(samplerate=samplerate, channels=data.shape[1], dtype=data.dtype) as stream:
        for start in range(0, len(data), BLOCK_FRAMES):
            if stop_event is not None and stop_event.is_set():
                return False
            stream.write(np.ascontiguousarray(data[start:start + BLOCK_FRAMES]))
    return True
