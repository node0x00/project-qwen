"""
Project Qwen - "what do you see" / "is anyone here" voice answers.

Runs object detection once on demand (YOLO26n via object_detector.py, the
accurate 640 model) on the shared CameraManager's frame and speaks a summary.

Better-view behavior: the robot repositions and looks again only when that
could change the answer - there's a possible subject (a "maybe" detection,
CANDIDATE_CONF..LOOK_CONF) but nothing confident:
  - "is anyone here": a maybe-person and no confirmed person
  - "what do you see": maybe-objects and no confident object at all
A confidently detected subject that's merely cut off at the frame edge does
NOT trigger it (furniture nearly always touches an edge, so that made the
robot move on almost every question); the cut-off edge only decides which
way to move. Escalation, one step per attempt, at most MAX_ADJUSTMENTS, and it
stops as soon as the answer is confident:
  1. lift up/down (toward the cut-off edge; up by default)
  2. a small in-place turn toward the side the subject is on
  3. a short, slow forward move - only if the IR sensors are clear
It never reverses (no rear sensors). Movement is skipped entirely when
allow_movement() is False (e.g. security mode).

The subject is the most likely "maybe" detection (person for "is anyone
here", any object for "what do you see").

Standalone test (needs the camera; robot on blocks if it may move):
    python scene_actions.py [--no-move]
"""

import time

LOOK_CONF = 0.35            # detections at/above this are reported
CANDIDATE_CONF = 0.25       # below LOOK_CONF but above this = "maybe" -> worth a better look
MAX_ADJUSTMENTS = 3
SETTLE_S = 0.8              # camera auto-exposure/motion settle after moving

LIFT_STEP_DEG = 20
LIFT_MIN_DEG, LIFT_MAX_DEG = 0, 130
TURN_SPEED, TURN_S = 70, 0.25        # small nudge, ~20 deg
FORWARD_SPEED, FORWARD_S = 50, 0.5


class SceneActions:
    def __init__(self, camera, tts, link=None, allow_movement=lambda: True, detector=None):
        """link: ESP32Link for repositioning (None = never move).
        detector: an object_detector.Detector; built lazily if None."""
        self.camera = camera
        self.tts = tts
        self.link = link
        self.allow_movement = allow_movement
        self._detector = detector

    # ------------------------------------------------------------------ voice entry points

    def describe_scene(self):
        detections = self._look(only_label=None)
        if detections is None:
            return
        from object_detector import describe
        self.tts.speak(describe(detections))

    def anyone_here(self):
        detections = self._look(only_label="person")
        if detections is None:
            return
        n = sum(1 for d in detections if d.label == "person")
        if n == 0:
            self.tts.speak("I don't see anyone.")
        elif n == 1:
            self.tts.speak("Yes, I see one person.")
        else:
            self.tts.speak(f"Yes, I see {n} people.")

    # ------------------------------------------------------------------ internals

    def _detector_ready(self):
        if self._detector is None:
            from object_detector import get_detector
            self._detector = get_detector(conf=CANDIDATE_CONF)
        return self._detector

    def _frame(self):
        for _ in range(20):
            frame = self.camera.get_frame()
            if frame is not None:
                return frame
            time.sleep(0.1)
        return None

    def _look(self, only_label):
        """Detects, repositions up to MAX_ADJUSTMENTS times if needed, and
        returns the confident detections (or None after speaking an error)."""
        try:
            detector = self._detector_ready()
        except Exception as e:
            print(f"[scene] detector unavailable: {e}")
            self.tts.speak("Sorry, my object detection isn't working.")
            return None

        announced = False
        tried_lift_dirs = set()
        detections = []
        for attempt in range(MAX_ADJUSTMENTS + 1):
            frame = self._frame()
            if frame is None:
                self.tts.speak("Sorry, I can't see anything, my camera isn't working.")
                return None
            all_dets = detector.detect(frame)       # includes "maybe" candidates
            from vision_overlay import overlay
            overlay.publish_objects(all_dets, source="scene")
            detections = [d for d in all_dets if d.confidence >= LOOK_CONF]
            relevant = [d for d in all_dets if only_label is None or d.label == only_label]
            subject = self._needs_better_view(relevant)
            if subject is None or attempt == MAX_ADJUSTMENTS or not self._can_move():
                break
            if not announced:
                self.tts.speak("Let me get a better look.")
                announced = True
            if not self._reposition(attempt, subject, frame.shape, tried_lift_dirs):
                break                                 # nothing left to try
            time.sleep(SETTLE_S)
        return detections

    @staticmethod
    def _needs_better_view(relevant):
        """The most likely "maybe" detection if there's no confident one -
        i.e. a better look could change the answer - else None."""
        if any(d.confidence >= LOOK_CONF for d in relevant):
            return None
        maybes = [d for d in relevant if d.confidence >= CANDIDATE_CONF]
        return max(maybes, key=lambda d: d.confidence) if maybes else None

    def _can_move(self) -> bool:
        return self.link is not None and self.link.connected and self.allow_movement()

    def _reposition(self, attempt, subject, frame_shape, tried_lift_dirs) -> bool:
        """One escalation step. Returns False if nothing could be done."""
        edges = subject.cut_off_edges(frame_shape)
        h, w = frame_shape[:2]
        steps = ("lift", "turn", "forward")
        for step in steps[min(attempt, len(steps) - 1):]:
            if step == "lift":
                direction = "down" if "bottom" in edges and "top" not in edges else "up"
                if direction in tried_lift_dirs:
                    continue
                pos = self.link.state.lift_pos or 0
                target = pos + LIFT_STEP_DEG if direction == "up" else pos - LIFT_STEP_DEG
                target = max(LIFT_MIN_DEG, min(LIFT_MAX_DEG, target))
                if target == pos:
                    continue
                tried_lift_dirs.add(direction)
                print(f"[scene] better view: lift {direction} to {target}")
                self.link.lift(target)
                self.link.wait_lift(5.0)
                return True
            if step == "turn":
                cx = subject.center[0]
                if abs(cx - w / 2) < w * 0.1 and not edges & {"left", "right"}:
                    continue                          # already centered - turning won't help
                left = "left" in edges or (cx < w / 2 and "right" not in edges)
                print(f"[scene] better view: nudge {'left' if left else 'right'}")
                speed = -TURN_SPEED if left else TURN_SPEED
                self.link.drive_for(speed, -speed, TURN_S)
                return True
            if step == "forward":
                if self.link.obstacle_ahead():
                    print("[scene] better view: forward skipped, obstacle ahead")
                    return False
                print("[scene] better view: short forward")
                self.link.drive_for(FORWARD_SPEED, FORWARD_SPEED, FORWARD_S)
                return True
        return False


if __name__ == "__main__":
    import sys
    from camera_manager import CameraManager

    class _PrintTTS:
        def speak(self, text):
            print(f"Qwen: {text}")

    camera = CameraManager()
    camera.start()
    link = None
    if "--no-move" not in sys.argv:
        from esp32_link import ESP32Link
        link = ESP32Link()
        link.start()
    actions = SceneActions(camera, _PrintTTS(), link)
    try:
        time.sleep(2)              # camera warm-up
        t = time.time()
        actions.describe_scene()
        print(f"(describe took {time.time() - t:.1f}s)")
        t = time.time()
        actions.anyone_here()
        print(f"(anyone_here took {time.time() - t:.1f}s)")
    finally:
        camera.stop()
        if link is not None:
            link.close()
