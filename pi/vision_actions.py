"""
Project Qwen - vision actions with verbal confirmation.

Both the voice command path (assistant.py) and the gesture path
(gesture_recognizer.py) call into this same class, so "start recording"
behaves identically and gets the same spoken confirmation regardless of
whether it came from your voice or a thumbs-up.

Security mode's trespasser response (security_mode.py) deliberately does
NOT go through this class - it calls CameraManager's start_recording()/
stop_recording() directly. Its spoken lines (the trespasser warning, not
"Recording started.") are different from the confirmations here, and mixing
the two would produce confusing/duplicate TTS output. This is intentional,
not an inconsistency to "fix" by routing it through VisionActions.
"""


class VisionActions:
    def __init__(self, camera_manager, tts, flashlight=None, notifier=None):
        """notifier: telegram_notify.TelegramNotifier - photos/videos go to the
        owner's phone and are deleted from the Pi once delivered. Without one
        (or if it isn't configured) they stay in captures/ as before."""
        self.camera_manager = camera_manager
        self.tts = tts
        self.flashlight = flashlight
        self.notifier = notifier

    def _to_phone(self) -> bool:
        return self.notifier is not None and self.notifier.enabled

    def start_recording(self):
        if self.camera_manager.is_recording():
            self.tts.speak("Already recording.")
            return
        path = self.camera_manager.start_recording()
        if path:
            self.tts.speak("Recording started.")
        else:
            self.tts.speak("Sorry, I couldn't start recording.")

    def stop_recording(self):
        if not self.camera_manager.is_recording():
            self.tts.speak("I wasn't recording.")
            return
        path = self.camera_manager.stop_recording()
        if path and self._to_phone():
            self.notifier.notify_video(path, "🎥 Your video")
            self.tts.speak("Recording stopped. Sending it to your phone.")
        else:
            self.tts.speak("Recording stopped.")

    def take_photo(self):
        path = self.camera_manager.take_photo()
        if path and self._to_phone():
            self.notifier.send_photo_file(path, "📸 Your photo")
            self.tts.speak("Got it, sending it to your phone.")
        elif path:
            self.tts.speak("Got it.")
        else:
            self.tts.speak("Sorry, I couldn't take the photo.")

    def flashlight_on(self):
        if self.flashlight is None:
            self.tts.speak("I don't have a flashlight.")
            return
        if self.flashlight.is_on:
            self.tts.speak("The light's already on.")
            return
        self.flashlight.on()
        self.tts.speak("Flashlight on.")

    def flashlight_off(self):
        if self.flashlight is None:
            self.tts.speak("I don't have a flashlight.")
            return
        if not self.flashlight.is_on:
            self.tts.speak("The light's already off.")
            return
        self.flashlight.off()
        self.tts.speak("Flashlight off.")

    def toggle_flashlight(self):
        if self.flashlight is not None and self.flashlight.is_on:
            self.flashlight_off()
        else:
            self.flashlight_on()


if __name__ == "__main__":
    from camera_manager import CameraManager

    class _PrintTTS:
        def speak(self, text: str):
            print(f"Qwen: {text}")

    from flashlight import get_flashlight

    print("Testing VisionActions standalone. Commands: r = start recording, "
          "s = stop recording, p = take a photo, l = toggle flashlight, q = quit.")

    camera = CameraManager()
    camera.start()
    actions = VisionActions(camera, _PrintTTS(), get_flashlight())
    try:
        while True:
            cmd = input("> ").strip().lower()
            if cmd == "r":
                actions.start_recording()
            elif cmd == "s":
                actions.stop_recording()
            elif cmd == "p":
                actions.take_photo()
            elif cmd == "l":
                actions.toggle_flashlight()
            elif cmd == "q":
                break
            else:
                print("Unknown command - use r/s/p/l/q.")
    finally:
        if actions.flashlight is not None:
            actions.flashlight.off()
        camera.stop()
