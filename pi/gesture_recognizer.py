"""
Project Qwen - gesture recognition.

Detects, from the webcam feed:
  - thumbs up (not currently recording) -> start recording
  - OK sign (while recording)           -> stop recording
  - peace sign, held, then a 1.5s delay -> take a photo

This module owns the camera by itself for now (standalone, testable on its
own). Once we wire in voice-triggered recording/photos, camera ownership
will move to a shared CameraManager so gesture watching and command
recording/photos don't fight over the same device - noted inline below.

Uses MediaPipe's Hand Landmarker task (21 landmarks/hand). Gestures are
classified from landmark geometry rather than a trained gesture model, which
keeps this cheap enough to run continuously on a Pi 4 while the robot is
"awake".

Run standalone to test detection against your own hand before wiring it
into the rest of the robot:
    python gesture_recognizer.py
"""

import collections
import math
import os
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

from vision_overlay import overlay

# Standard 21-point hand skeleton topology (wrist + 4 joints per finger).
# Hardcoded rather than pulled from mp.solutions.hands.HAND_CONNECTIONS,
# since that legacy API was removed in newer mediapipe versions - this
# topology itself never changes, so hardcoding it is version-independent.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # wrist to pinky base
]

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "hand_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

# Piper voice, only loaded when this file is run standalone (see __main__) -
# same model assistant.py uses, so spoken confirmations sound identical.
PIPER_MODEL_PATH = os.path.join(MODEL_DIR, "en_US-lessac-medium.onnx")

CAMERA_INDEX = 0          # adjust if the C920 isn't device 0 on your system
TARGET_FPS = 12           # gesture watching doesn't need full 30fps - saves CPU
MIN_HAND_DETECTION_CONFIDENCE = 0.5
MIN_HAND_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# How long a classified gesture must hold steady before it "counts" -
# filters out single-frame misclassifications/flicker.
GESTURE_HOLD_S = 0.15

# Peace-sign-to-photo delay
PHOTO_DELAY_S = 1.5

# Cooldown after any triggered action, so the same held gesture doesn't
# immediately re-trigger the next action.
ACTION_COOLDOWN_S = 2.0

# How long the "STOP RECORDING" / "TAKE PHOTO" flash message stays on screen.
MESSAGE_FLASH_S = 1.5

# Thumb-extended threshold: cosine of the bend angle at the thumb's IP
# joint. Above this = straight/extended; below = bent (tucked into a fist).
# Tune against the live value shown in the debug window if fist/open aren't
# classifying correctly for your hand.
THUMB_STRAIGHTNESS_THRESHOLD = 0.7

# Distinguishes "thumbs up" from "fist": how far above the folded
# fingertips the thumb tip sits, normalized by palm size. Many people's
# fist keeps the thumb fairly straight (it just lies across the curled
# fingers) so straightness alone can't tell the two apart - this checks
# whether the thumb is actually raised clear of the folded fingers.
THUMB_RAISED_RATIO_THRESHOLD = 1.25

# OK sign: thumb tip and index tip touching, normalized by palm size.
OK_SIGN_TOUCH_RATIO = 0.4

# Rock/horns sign (index + pinky extended, middle + ring curled) toggles the
# flashlight once held this long. Longer than GESTURE_HOLD_S on purpose -
# toggling a light by accident is more annoying than a brief wait.
ROCK_HOLD_S = 1.0
# Brief classification dropouts (a frame or two of "unknown") shorter than
# this don't reset the hold timer.
ROCK_DROPOUT_S = 0.6       # at ~5-10 fps one misread frame is already a 0.1-0.2 s gap
# The spec says "others curled" (thumb tucked). Set True to also accept the
# thumb sticking out (the "I love you" sign) if your rock sign isn't detected.
ROCK_ALLOW_THUMB_EXTENDED = True    # most people make the sign with the thumb out
# "L" sign (index up, thumb out to the side, other fingers curled) locks the door.
# Held like the rock sign, so a passing hand can't lock anyone out.
L_HOLD_S = 1.0
L_THUMB_ANGLE_DEG = 45     # thumb at least this far from the index direction
L_THUMB_REACH = 0.6        # thumb tip at least this far (x palm size) from the index knuckle

# Hand landmark indices (MediaPipe convention)
WRIST = 0
THUMB_TIP, THUMB_IP, THUMB_MCP = 4, 3, 2
INDEX_TIP, INDEX_PIP, INDEX_MCP = 8, 6, 5
MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP = 12, 10, 9
RING_TIP, RING_PIP, RING_MCP = 16, 14, 13
PINKY_TIP, PINKY_PIP, PINKY_MCP = 20, 18, 17


# --------------------------------------------------------------------------
# Model download
# --------------------------------------------------------------------------

def ensure_model_downloaded():
    if os.path.exists(MODEL_PATH):
        return
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"Downloading hand_landmarker.task (~12MB) to {MODEL_PATH}...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete.")


# --------------------------------------------------------------------------
# Gesture classification from landmark geometry
# --------------------------------------------------------------------------

def _dist(a, b) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def _finger_extended(landmarks, tip_idx, pip_idx, mcp_idx) -> bool:
    """A finger counts as extended if its tip is meaningfully farther from
    the wrist than its own PIP joint - this is fairly rotation-tolerant
    since it's a distance ratio, not just a y-coordinate comparison."""
    wrist = landmarks[WRIST]
    tip_dist = _dist(wrist, landmarks[tip_idx])
    pip_dist = _dist(wrist, landmarks[pip_idx])
    return tip_dist > pip_dist * 1.15


def _thumb_straightness(landmarks) -> float:
    """Cosine of the angle between the thumb's two segments (MCP->IP and
    IP->TIP). Near 1.0 = straight thumb (extended); drops noticeably when
    the thumb bends at the IP joint (curled into a fist). Using straightness
    instead of tip-distance-from-hand means this doesn't care whether the
    thumb is abducted out to the side or held close alongside the hand -
    both count as "extended" as long as the thumb itself isn't bent."""
    mcp, ip, tip = landmarks[THUMB_MCP], landmarks[THUMB_IP], landmarks[THUMB_TIP]
    v1 = (ip.x - mcp.x, ip.y - mcp.y)
    v2 = (tip.x - ip.x, tip.y - ip.y)
    mag1, mag2 = math.hypot(*v1), math.hypot(*v2)
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return (v1[0] * v2[0] + v1[1] * v2[1]) / (mag1 * mag2)


def _thumb_extended(landmarks) -> bool:
    return _thumb_straightness(landmarks) > THUMB_STRAIGHTNESS_THRESHOLD


def _thumb_raised_ratio(landmarks) -> float:
    """How far above the folded fingertips the thumb tip sits (positive =
    thumb higher / y smaller than the folded fingers), normalized by palm
    size. Large positive = thumbs-up; near zero or negative = thumb lying
    across the fingers as in a normal fist."""
    folded_tips_y = [landmarks[i].y for i in (INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)]
    avg_folded_y = sum(folded_tips_y) / len(folded_tips_y)
    palm_scale = _dist(landmarks[WRIST], landmarks[MIDDLE_MCP])
    return (avg_folded_y - landmarks[THUMB_TIP].y) / palm_scale


def _thumb_index_touching(landmarks) -> bool:
    """True when thumb tip and index tip are close together (the pinch/circle
    shape of an OK sign), normalized by palm size."""
    palm_scale = _dist(landmarks[WRIST], landmarks[MIDDLE_MCP])
    return _dist(landmarks[THUMB_TIP], landmarks[INDEX_TIP]) < palm_scale * OK_SIGN_TOUCH_RATIO


def _thumb_spread(landmarks) -> bool:
    """Thumb pointing well away from the index finger (the corner of an L)."""
    t = (landmarks[THUMB_TIP].x - landmarks[THUMB_MCP].x, landmarks[THUMB_TIP].y - landmarks[THUMB_MCP].y)
    i = (landmarks[INDEX_TIP].x - landmarks[INDEX_MCP].x, landmarks[INDEX_TIP].y - landmarks[INDEX_MCP].y)
    mt, mi = math.hypot(*t), math.hypot(*i)
    if mt == 0 or mi == 0:
        return False
    angle = math.degrees(math.acos(max(-1.0, min(1.0, (t[0] * i[0] + t[1] * i[1]) / (mt * mi)))))
    palm_scale = _dist(landmarks[WRIST], landmarks[MIDDLE_MCP])
    reach = _dist(landmarks[THUMB_TIP], landmarks[INDEX_MCP]) / palm_scale
    return angle >= L_THUMB_ANGLE_DEG and reach >= L_THUMB_REACH


def classify_gesture(landmarks) -> str:
    """Returns one of: 'fist', 'open', 'peace', 'thumbs_up', 'ok_sign', 'rock', 'L', 'unknown'."""
    thumb_straight = _thumb_extended(landmarks)
    index = _finger_extended(landmarks, INDEX_TIP, INDEX_PIP, INDEX_MCP)
    middle = _finger_extended(landmarks, MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP)
    ring = _finger_extended(landmarks, RING_TIP, RING_PIP, RING_MCP)
    pinky = _finger_extended(landmarks, PINKY_TIP, PINKY_PIP, PINKY_MCP)

    four_fingers_curled = not (index or middle or ring or pinky)
    four_fingers_extended = index and middle and ring and pinky
    three_extended_no_index = middle and ring and pinky and not index

    if three_extended_no_index and _thumb_index_touching(landmarks):
        return "ok_sign"
    if four_fingers_curled:
        if _thumb_raised_ratio(landmarks) > THUMB_RAISED_RATIO_THRESHOLD:
            return "thumbs_up"
        return "fist"
    if thumb_straight and four_fingers_extended:
        return "open"
    if index and middle and not ring and not pinky:
        return "peace"
    if index and pinky and not middle and not ring and \
            (ROCK_ALLOW_THUMB_EXTENDED or not thumb_straight):
        return "rock"
    if index and not middle and not ring and not pinky and thumb_straight and _thumb_spread(landmarks):
        return "L"
    return "unknown"


def draw_landmarks(frame, landmarks):
    """Draws the hand skeleton onto `frame` in place - call this BEFORE
    mirroring for display, so landmark positions line up with the frame
    they were computed from."""
    h, w = frame.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (255, 255, 255), 2)

    for i, (x, y) in enumerate(points):
        if i == THUMB_TIP:
            color = (0, 0, 255)      # red - thumb tip
        elif i in (INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP):
            color = (0, 255, 255)    # yellow - other fingertips
        else:
            color = (0, 255, 0)      # green - other joints
        cv2.circle(frame, (x, y), 5, color, -1)


# --------------------------------------------------------------------------
# Gesture sequence / hold state machines
# --------------------------------------------------------------------------

class GestureStateMachine:
    """
    Tracks a debounced stream of classified gestures and fires callbacks for:
      - thumbs_up (not currently recording) => on_start_recording()
      - ok_sign (while recording)           => on_stop_recording()
      - peace held, then +1.5s             => on_take_photo()
      - rock held for ROCK_HOLD_S          => on_toggle_flashlight()
      - L held for L_HOLD_S                => on_lock_door()
    """

    def __init__(self, on_start_recording, on_stop_recording, on_take_photo,
                 is_recording_fn, on_toggle_flashlight=None, on_lock_door=None):
        self.on_start_recording = on_start_recording
        self.on_stop_recording = on_stop_recording
        self.on_take_photo = on_take_photo
        self.is_recording_fn = is_recording_fn
        self.on_toggle_flashlight = on_toggle_flashlight
        self.on_lock_door = on_lock_door
        self._l_since = None
        self._l_last_seen = 0.0
        self._l_fired = False

        self._rock_since = None       # when the current rock hold started
        self._rock_last_seen = 0.0
        self._rock_fired = False      # one toggle per hold; release to toggle again

        self._raw_history = collections.deque(maxlen=5)  # for debouncing
        self._stable_gesture = None
        self._stable_since = 0.0

        self._photo_timer_deadline = None
        self._last_action_at = 0.0

    def _debounce(self, raw_gesture: str, now: float) -> str | None:
        """Returns a gesture only once it's been stable for GESTURE_HOLD_S,
        and only once per stable period (so it doesn't fire every frame)."""
        self._raw_history.append(raw_gesture)
        if raw_gesture != self._stable_gesture:
            self._stable_gesture = raw_gesture
            self._stable_since = now
            return None
        if now - self._stable_since >= GESTURE_HOLD_S and self._stable_since != -1:
            self._stable_since = -1  # mark as "already reported"
            return raw_gesture
        return None

    def _rock_held(self, raw_gesture: str, now: float, can_fire: bool) -> bool:
        """True once per continuous rock hold of ROCK_HOLD_S, tolerating
        dropouts shorter than ROCK_DROPOUT_S. Tracks the hold even while
        can_fire is False (action cooldown), so a release + re-raise during
        the cooldown is still seen."""
        if raw_gesture == "rock":
            if self._rock_since is None:
                self._rock_since = now
            self._rock_last_seen = now
        elif self._rock_since is not None and now - self._rock_last_seen > ROCK_DROPOUT_S:
            self._rock_since = None
            self._rock_fired = False
        if self._rock_since is None or self._rock_fired or not can_fire:
            return False
        if self._rock_last_seen - self._rock_since >= ROCK_HOLD_S:
            self._rock_fired = True
            return True
        return False

    def _l_held(self, raw_gesture: str, now: float, can_fire: bool) -> bool:
        """Like _rock_held: true once per continuous L hold of L_HOLD_S."""
        if raw_gesture == "L":
            if self._l_since is None:
                self._l_since = now
            self._l_last_seen = now
        elif self._l_since is not None and now - self._l_last_seen > ROCK_DROPOUT_S:
            self._l_since = None
            self._l_fired = False
        if self._l_since is None or self._l_fired or not can_fire:
            return False
        if self._l_last_seen - self._l_since >= L_HOLD_S:
            self._l_fired = True
            return True
        return False

    def rock_hold_progress(self) -> float | None:
        """0..1 progress of the current rock hold, for the debug overlay."""
        if self._rock_since is None or self._rock_fired:
            return None
        return min(1.0, (self._rock_last_seen - self._rock_since) / ROCK_HOLD_S)

    def update(self, raw_gesture: str):
        now = time.time()

        cooling_down = now - self._last_action_at < ACTION_COOLDOWN_S
        rock_fire = (self.on_toggle_flashlight is not None
                     and self._rock_held(raw_gesture, now, can_fire=not cooling_down))

        l_fire = (self.on_lock_door is not None
                  and self._l_held(raw_gesture, now, can_fire=not cooling_down))

        if cooling_down:
            return  # still cooling down from the last triggered action

        if l_fire:
            self.on_lock_door()
            self._last_action_at = now
            self._photo_timer_deadline = None
            return

        if rock_fire:
            self.on_toggle_flashlight()
            self._last_action_at = now
            self._photo_timer_deadline = None
            return

        stable = self._debounce(raw_gesture, now)

        if stable == "thumbs_up" and not self.is_recording_fn():
            self.on_start_recording()
            self._last_action_at = now
            self._photo_timer_deadline = None
            return

        if stable == "ok_sign" and self.is_recording_fn():
            self.on_stop_recording()
            self._last_action_at = now
            self._photo_timer_deadline = None
            return

        # --- peace sign -> 1.5s delay -> photo ---
        if stable == "peace" and self._photo_timer_deadline is None:
            self._photo_timer_deadline = now + PHOTO_DELAY_S

        if self._photo_timer_deadline is not None and now >= self._photo_timer_deadline:
            self.on_take_photo()
            self._last_action_at = now
            self._photo_timer_deadline = None


# --------------------------------------------------------------------------
# Main gesture watcher
# --------------------------------------------------------------------------

class GestureController:
    def __init__(self, camera_manager, on_start_recording, on_stop_recording, on_take_photo,
                 is_recording_fn, on_toggle_flashlight=None, on_lock_door=None):
        ensure_model_downloaded()
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=MIN_HAND_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=MIN_HAND_PRESENCE_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        )
        self.landmarker = HandLandmarker.create_from_options(options)
        self.camera_manager = camera_manager
        self.is_recording_fn = is_recording_fn

        self._message = None
        self._message_until = 0.0

        def _flash(text):
            self._message = text
            self._message_until = time.time() + MESSAGE_FLASH_S

        def _wrapped_stop_recording():
            _flash("STOP RECORDING")
            on_stop_recording()

        def _wrapped_take_photo():
            _flash("TAKE PHOTO")
            on_take_photo()

        def _wrapped_toggle_flashlight():
            _flash("FLASHLIGHT")
            on_toggle_flashlight()

        def _wrapped_lock_door():
            _flash("LOCK DOOR")
            on_lock_door()

        # Set while the voice pipeline is busy (wake word -> reply) so hand
        # tracking doesn't compete with speech-to-text / the LLM for the CPU.
        self._paused = False

        self.state_machine = GestureStateMachine(
            on_start_recording, _wrapped_stop_recording, _wrapped_take_photo, is_recording_fn,
            on_toggle_flashlight=_wrapped_toggle_flashlight if on_toggle_flashlight else None,
            on_lock_door=_wrapped_lock_door if on_lock_door else None,
        )
        self._frame_ms = int(1000 / TARGET_FPS)

    def set_paused(self, paused: bool):
        self._paused = paused

    def run_forever(self, show_debug_window: bool = False, stop_event=None):
        """Blocking loop - call from a dedicated thread alongside the rest
        of the robot's control loop. Reads frames from the shared
        CameraManager rather than owning a camera device itself. If
        `stop_event` is given, the loop exits cleanly once it's set
        (checked once per frame) - used by security mode to stop gesture
        watching without killing the whole thread forcibly."""
        from stream_server import lower_thread_priority
        lower_thread_priority()      # never starve the wake-word listener
        try:
            while stop_event is None or not stop_event.is_set():
                loop_start = time.time()
                if self._paused:
                    time.sleep(0.1)
                    continue
                frame = self.camera_manager.get_frame()
                if frame is None:
                    time.sleep(0.05)
                    continue

                # MediaPipe VIDEO mode needs timestamps that never go backwards - even
                # across restarts of this loop (security mode stops/starts gestures),
                # so use a monotonic clock, not time since this run started.
                timestamp_ms = int(time.monotonic() * 1000)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

                gesture = "unknown"
                thumb_straightness = None
                thumb_raised_ratio = None
                landmarks = None
                if result.hand_landmarks:
                    landmarks = result.hand_landmarks[0]
                    gesture = classify_gesture(landmarks)
                    thumb_straightness = _thumb_straightness(landmarks)
                    thumb_raised_ratio = _thumb_raised_ratio(landmarks)

                self.state_machine.update(gesture)
                overlay.publish_hands([(landmarks, gesture)] if landmarks is not None else [])

                if show_debug_window:
                    # Draw the skeleton on the ORIGINAL (unflipped) frame first,
                    # so landmark pixel positions line up correctly, then mirror
                    # the whole annotated frame for a natural "selfie" view.
                    annotated = frame.copy()
                    if landmarks is not None:
                        draw_landmarks(annotated, landmarks)
                    display_frame = cv2.flip(annotated, 1)

                    cv2.putText(display_frame, gesture, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                                2, (0, 255, 0), 2)
                    if thumb_straightness is not None:
                        cv2.putText(display_frame, f"thumb_straightness={thumb_straightness:.2f}",
                                    (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 2)
                        cv2.putText(display_frame, f"thumb_raised_ratio={thumb_raised_ratio:.2f}",
                                    (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 2)

                    # Rock-sign hold progress bar (flashlight toggles when full).
                    rock_progress = self.state_machine.rock_hold_progress()
                    if rock_progress is not None:
                        h = display_frame.shape[0]
                        cv2.rectangle(display_frame, (10, h - 40), (310, h - 15), (255, 255, 255), 2)
                        cv2.rectangle(display_frame, (10, h - 40),
                                      (10 + int(300 * rock_progress), h - 15), (0, 255, 255), -1)

                    # "RECORDING" indicator, top-right, only while actually recording.
                    if self.is_recording_fn():
                        h, w = display_frame.shape[:2]
                        text = "RECORDING"
                        (text_w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                        cv2.putText(display_frame, text, (w - text_w - 20, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    # "STOP RECORDING" / "TAKE PHOTO" flash message, centered.
                    if self._message is not None and time.time() < self._message_until:
                        h, w = display_frame.shape[:2]
                        (text_w, text_h), _ = cv2.getTextSize(
                            self._message, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
                        x, y = (w - text_w) // 2, (h + text_h) // 2
                        cv2.putText(display_frame, self._message, (x, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

                    cv2.imshow("Gesture debug", display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                # Throttle to TARGET_FPS - gesture watching doesn't need max speed
                # Sleep only what's left of the frame budget (it used to sleep the
                # whole budget on top of ~100 ms of processing -> ~5 fps).
                time.sleep(max(0.0, self._frame_ms / 1000 - (time.time() - loop_start)))
        finally:
            if show_debug_window:
                cv2.destroyAllWindows()


# --------------------------------------------------------------------------
# Standalone test
# --------------------------------------------------------------------------

if __name__ == "__main__":
    from camera_manager import CameraManager
    from vision_actions import VisionActions

    class _SpeakerTTS:
        """Real Piper voice output, lazy-loaded on first use, so this file
        can be run standalone (camera + gestures only) while still giving
        VisionActions actual spoken confirmation - not just a print. Skips
        assistant.py's mic-muting logic since no mic is being listened to
        here."""

        def __init__(self):
            self._voice = None

        def _ensure_loaded(self):
            if self._voice is None:
                from piper.voice import PiperVoice
                print("Loading Piper voice...")
                self._voice = PiperVoice.load(PIPER_MODEL_PATH)

        def speak(self, text: str):
            if not text.strip():
                return
            self._ensure_loaded()
            print(f"Qwen: {text}")

            import numpy as np
            import sounddevice as sd

            audio_chunks = []
            sample_rate = None
            for chunk in self._voice.synthesize(text):
                sample_rate = chunk.sample_rate
                audio_chunks.append(np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16))
            if not audio_chunks:
                return
            audio = np.concatenate(audio_chunks)
            sd.play(audio, samplerate=sample_rate)
            sd.wait()

    from flashlight import get_flashlight

    print("Testing gesture recognition. Try: thumbs up, OK sign, peace sign, rock sign (1s).")
    print("Photos/videos save to captures/photos and captures/videos, same as assistant.py.")
    print("Press 'q' in the debug window to quit.\n")

    camera = CameraManager(camera_index=CAMERA_INDEX)
    camera.start()
    flashlight = get_flashlight()
    vision_actions = VisionActions(camera, _SpeakerTTS(), flashlight)

    controller = GestureController(
        camera_manager=camera,
        on_start_recording=vision_actions.start_recording,
        on_stop_recording=vision_actions.stop_recording,
        on_take_photo=vision_actions.take_photo,
        is_recording_fn=camera.is_recording,
        on_toggle_flashlight=vision_actions.toggle_flashlight,
    )
    try:
        controller.run_forever(show_debug_window=True)
    finally:
        flashlight.off()
        camera.stop()
