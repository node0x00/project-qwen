"""
Project Qwen - security mode state machine.

States:
  OFF          - normal operation (voice + gestures), the default. Faces are
                 never classified here: an unknown face does nothing.
  ARMING       - exit delay (EXIT_DELAY_S) after arming, so the owner can
                 leave without tripping the motion sensor.
  ARMED_IDLE   - everything off (mic/wake word, camera, gestures, LLM, room
                 lights) except a lightweight PIR motion listener.
  ARMED_ACTIVE - camera + face recognition + throttled person detection.
                 Voice is back on; gestures stay off; voice recording/photo/
                 movement commands are suppressed.

Arming: "Hey Qwen, security mode", or keypad D (exit button) via the
smart-home hub (smart_home.py).

Two ways into ARMED_ACTIVE:
  - Motion (PIR): room lights on (hub), camera on. No detection for
    NO_DETECTION_TIMEOUT_S -> lights off, back to ARMED_IDLE.
  - Remote unlock (web page UNLOCK) - in ANY state, including OFF: security
    mode turns on (no exit delay) and stays on until the owner is verified.
    The hub already unlocked the door, turned the lights on and started the
    keypad's 3-minute countdown. Qwen raises the lift to REMOTE_LIFT_DEG
    immediately (faces are easier to see from up there - the robot sits on
    the floor) and looks for the owner for REMOTE_UNLOCK_WINDOW_S. If the window runs out without verification, the
    alarm sounds (capped at ALARM_MAX_S) and security mode stays armed.

Disarm (-> OFF, lift back to 0, "Welcome back"):
  - owner's face recognized (OWNER_CONFIRM_S). During a remote unlock this
    also cancels the keypad countdown and locks the door, and it wins even
    if an unknown face is in view too. Outside a remote unlock, an active
    trespasser response still blocks the owner exit (original rule).
  - correct keypad code (keypad already stopped its countdown and locked the door).

A face only counts as "unknown" once it's been seen for UNKNOWN_CONFIRM_S,
is at least MIN_UNKNOWN_FACE_PX wide (tiny distant faces score badly), and
the owner isn't recognized as the only face in the frame - so the owner's
face flickering between owner/unknown never sets off the warning.

Unknown face while ARMED_ACTIVE starts an incident: recording + spoken
warning, alarm ALARM_TRIGGER_S after the first sighting (REMOTE_ALARM_TRIGGER_S
during a remote unlock). The intruder counts as present while EITHER an
unknown face OR a person (YOLO, works when they turn away) is seen; the
incident ends after INCIDENT_CLEAR_S with neither. Outside a remote unlock the
door is unlocked (so they can leave rather than be trapped) and locked again
only after DOOR_RELOCK_CLEAR_S with nobody seen; during a remote unlock the
door is left as the hub/keypad have it. In testing, tying this to the face
alone ended the incident whenever the face turned away for a few seconds -
the door relocked after 9 s and the 25 s alarm never fired.

A person seen without an unknown face (e.g. facing away) records silently
and keeps the no-detection timer alive - person detection can't tell the
owner from a stranger, only faces can.

"Hey Qwen, alarm" sounds the alarm for VOICE_ALARM_S in any state, and
nothing (not even the owner's face) stops it early.

Warning + alarm play at SECURITY_VOLUME (system volume restored afterwards).

Push notifications (Telegram, telegram_notify.py) go out once per incident,
never per frame: armed, motion (photo), person (photo, at most every
PERSON_NOTIFY_INTERVAL_S), unknown person (photo), remote unlock started /
verified / not verified, alarm (photo), disarmed.

This module owns none of the shared subsystems - it's driven via callbacks
from assistant.py's main(), so it stays testable without the full stack.
"""

import enum
import subprocess
import threading
import time

from alarm import Alarm
from vision_overlay import overlay
from face_recognizer import FaceRecognizer
from pir_sensor import get_pir_sensor


class SecurityState(enum.Enum):
    OFF = "off"
    ARMING = "arming"
    ARMED_IDLE = "armed_idle"
    ARMED_ACTIVE = "armed_active"


ARMED_STATES = (SecurityState.ARMING, SecurityState.ARMED_IDLE, SecurityState.ARMED_ACTIVE)

# Exit delay between arming and the motion sensor counting.
EXIT_DELAY_S = 100

# How often the face-recognition loop polls a frame while ARMED_ACTIVE.
# Heavier than gesture watching (two DNN passes per detected face), so
# throttled more.
FACE_LOOP_FPS = 5

# Debounce: how long an unknown face may be briefly undetected before the
# trespasser response is considered "gone" - flicker in face detection is
# normal and shouldn't stop/start recording repeatedly. Tunable.
UNKNOWN_ABSENCE_GRACE_S = 4.0

# Incident timing (an incident = an unknown face was confirmed).
INCIDENT_CLEAR_S = 20.0             # nobody (no unknown face, no person) this long -> incident over
DOOR_RELOCK_CLEAR_S = 60.0          # nobody this long -> relock the door we unlocked
OWNER_FLICKER_S = 0.6               # owner confirmation tolerates detection gaps this short

# Alarm this long after the first unknown-face sighting, if the incident is still on.
ALARM_TRIGGER_S = 25.0
REMOTE_ALARM_TRIGGER_S = 20.0       # during a remote unlock
ALARM_MAX_S = 5 * 60                # any security alarm stops after this
VOICE_ALARM_S = 60                  # "Hey Qwen, alarm"

# If a trespasser encounter drags on, re-speak the warning at this interval.
TRESPASSER_WARNING_REPEAT_S = 30.0

# No face or person for this long while ARMED_ACTIVE -> assume a false PIR
# trigger, lights off, back to ARMED_IDLE. Not during a remote-unlock window.
NO_DETECTION_TIMEOUT_S = 120.0

# Small debounce before trusting an owner match enough to disarm, so a single
# misclassified frame can't flap the state machine.
OWNER_CONFIRM_S = 1.0

# Remote unlock: matches the keypad's own countdown.
REMOTE_UNLOCK_WINDOW_S = 180.0
BLINK_HALF_PERIOD_S = 0.125         # flashlight flashing during the alarm (4 Hz)
REMOTE_LIFT_DEG = 130
REMOTE_LIFT_MS = 1500               # straight up as soon as the remote unlock arrives
LIFT_HOME_DEG = 0
LIFT_MOVE_MS = 3000

# False-trespasser guard (the owner was briefly misread as "unknown" in testing).
UNKNOWN_CONFIRM_S = 1.5
# Faces narrower than this are too small to judge: the owner scored as low as
# 0.16-0.24 at ~80-95 px in testing (false alarm). Small faces still count as
# "someone is here", and a person farther away is still recorded silently.
MIN_UNKNOWN_FACE_PX = 110
FACE_LOG_INTERVAL_S = 1.0           # log face labels/scores at most this often (for tuning)

# Full-body person detection (YOLO26n, small 320 model) alongside face
# detection, so someone facing away from the camera still counts. Throttled -
# it's the heaviest thing in this loop.
PERSON_CHECK_FPS = 2.0
PERSON_MODEL_DIR = "models/yolo26n_320_ncnn_model"
PERSON_MODEL_IMGSZ = 320
PERSON_CONF = 0.45

# Warning + alarm volume. 0.50 while bench testing (neighbors!) - set to 1.0
# for the finished robot. Normal speech keeps the system volume.
SECURITY_VOLUME = 0.50

# Notifications
PERSON_NOTIFY_INTERVAL_S = 60.0
NOTIFY_PHOTO_DELAY_S = 1.5          # let the camera settle before the "motion" photo

WARNING_TEXT = "You are trespassing and being recorded, leave the premises immediately."


class _VolumeOverride:
    """Sets the speaker to SECURITY_VOLUME while any security sound is
    playing, and restores the previous volume after the last one ends."""

    def __init__(self, level: float):
        self.level = level
        self._count = 0
        self._saved = None
        self._lock = threading.Lock()

    @staticmethod
    def _wpctl(*args) -> str:
        try:
            return subprocess.run(["wpctl", *args], capture_output=True, text=True, timeout=3).stdout
        except Exception:
            return ""

    def acquire(self):
        with self._lock:
            if self._count == 0:
                out = self._wpctl("get-volume", "@DEFAULT_AUDIO_SINK@")   # "Volume: 0.30"
                try:
                    self._saved = float(out.split()[1])
                except (IndexError, ValueError):
                    self._saved = None
                self._wpctl("set-volume", "@DEFAULT_AUDIO_SINK@", str(self.level))
            self._count += 1

    def release(self):
        with self._lock:
            self._count = max(0, self._count - 1)
            if self._count == 0 and self._saved is not None:
                self._wpctl("set-volume", "@DEFAULT_AUDIO_SINK@", str(self._saved))


class SecurityController:
    def __init__(self, camera, tts, start_voice, stop_voice,
                 start_gestures, stop_gestures, start_camera, stop_camera,
                 hub=None, link=None, notifier=None, flashlight=None, tracker=None):
        """hub: smart_home.SmartHomeHub (lights/door/keypad), or None.
        link: esp32_link.ESP32Link (lift), or None.
        notifier: telegram_notify.TelegramNotifier (push alerts), or None."""
        self.camera = camera
        self.tts = tts
        self.hub = hub
        self.link = link
        self.notifier = notifier
        self.flashlight = flashlight    # head LED: on when someone is seen, flashing with the alarm
        self.tracker = tracker          # person_tracker.PersonTracker: turns to keep the person in view
        self._lit_up = False            # lights + flashlight already turned on for this ARMED-ACTIVE spell
        self._blink_stop = None
        self._start_voice = start_voice
        self._stop_voice = stop_voice
        self._start_gestures = start_gestures
        self._stop_gestures = stop_gestures
        self._start_camera = start_camera
        self._stop_camera = stop_camera
        self._camera_on = True          # assistant.py starts with the camera running
        self.remote_lift_pending = False

        self.state = SecurityState.OFF
        self._state_lock = threading.Lock()   # arm() vs. remote unlock arriving together
        self._face_recognizer = None    # lazy - only built on first ARMED_ACTIVE entry
        self._person_detector = None    # lazy too; False if unavailable
        self.alarm = Alarm()
        self._pir = None
        self._volume = _VolumeOverride(SECURITY_VOLUME)

        # Cross-thread signals (hub poller / voice thread -> security thread).
        self._remote_unlock = threading.Event()
        self._password_ok = threading.Event()
        self._countdown_expired = threading.Event()

        # Alarm ownership: security alarm vs. voice alarm share one Alarm.
        self._alarm_lock = threading.Lock()
        self._security_alarm_since = None
        self._voice_alarm_until = 0.0
        self._manual_alarm = False      # control-page alarm: on until the button is pressed again

    # ------------------------------------------------------------------ public API

    def is_armed_active(self) -> bool:
        return self.state == SecurityState.ARMED_ACTIVE

    def is_armed(self) -> bool:
        return self.state in ARMED_STATES

    def arm(self, source: str = "voice"):
        """OFF -> ARMING -> (after EXIT_DELAY_S) ARMED_IDLE."""
        with self._state_lock:
            if self.state != SecurityState.OFF:
                return
            print(f"[security] Armed by {source} - exit delay {EXIT_DELAY_S}s")
            self.state = SecurityState.ARMING
            self._remote_unlock.clear()
            self._password_ok.clear()
            self._countdown_expired.clear()
        self._notify(f"🔒 Security mode on ({source}). Armed in {EXIT_DELAY_S} s.")
        self.tts.speak(f"Security mode on. Arming in {EXIT_DELAY_S} seconds.")
        self._stop_gestures()
        self._stop_voice()
        self._camera_off()
        threading.Thread(target=self._run, daemon=True).start()

    def on_hub_event(self, event: str):
        """Called by SmartHomeHub's poller thread."""
        if event == "exit":
            self.arm(source="keypad")
        elif event == "remote_unlock":
            print("[security] Remote unlock -> lift up, verify owner")
            self.remote_lift_pending = True                 # assistant.py lets this one through while charging
            try:
                self._lift(REMOTE_LIFT_DEG, REMOTE_LIFT_MS)  # first, before anything slow
            finally:
                self.remote_lift_pending = False
            with self._state_lock:
                if self.state != SecurityState.OFF:
                    self._remote_unlock.set()               # the running security thread picks it up
                    return
                self.state = SecurityState.ARMED_ACTIVE     # security on right away, no exit delay
                self._remote_unlock.set()
                self._password_ok.clear()
                self._countdown_expired.clear()
            self._stop_gestures()
            threading.Thread(target=self._run, kwargs={"start": "remote"}, daemon=True).start()
        elif event == "password_ok":
            if self.is_armed():
                self._password_ok.set()
        elif event == "countdown_expired":
            if self.is_armed():
                self._countdown_expired.set()

    def voice_alarm(self):
        """'Hey Qwen, alarm': VOICE_ALARM_S of alarm that nothing stops early."""
        print(f"[security] Voice alarm for {VOICE_ALARM_S}s")
        self._notify(f"🚨 Alarm sounded by voice command ({VOICE_ALARM_S} s).")
        with self._alarm_lock:
            first = self._voice_alarm_until < time.time()
            self._voice_alarm_until = time.time() + VOICE_ALARM_S
            self._alarm_on()
        if first:
            threading.Thread(target=self._voice_alarm_timer, daemon=True).start()

    def toggle_manual_alarm(self) -> bool:
        """Control-page alarm button: on until pressed again. Returns the new state."""
        with self._alarm_lock:
            self._manual_alarm = not self._manual_alarm
            if self._manual_alarm:
                print("[security] Alarm ON from the control page")
                self._alarm_on()
            elif not self._other_alarm_owner():
                print("[security] Alarm OFF from the control page")
                self._alarm_off()
            on = self._manual_alarm
        self._notify("🚨 Alarm switched on from the control page." if on else "🔕 Alarm switched off from the control page.")
        return on

    def manual_alarm_on(self) -> bool:
        return self._manual_alarm

    def _other_alarm_owner(self) -> bool:
        # Caller holds _alarm_lock.
        since = self._security_alarm_since
        return time.time() < self._voice_alarm_until or (since is not None and since != float("inf"))

    def shutdown(self):
        self.state = SecurityState.OFF
        self._manual_alarm = False
        with self._alarm_lock:
            self._security_alarm_since = None
            self._voice_alarm_until = 0.0
            self._alarm_off()

    # ------------------------------------------------------------------ alarm ownership

    def _alarm_on(self):
        # Caller holds _alarm_lock.
        if not self.alarm.is_playing():
            self._volume.acquire()
            self.alarm.start()
            self._lights(True)          # room lights stay on for the whole alarm
            self._start_blink()

    def _alarm_off(self):
        # Caller holds _alarm_lock.
        if self.alarm.is_playing():
            self.alarm.stop()
            self._volume.release()
            self._stop_blink()

    # ------------------------------------------------------------------ flashlight

    def _flash(self, on: bool):
        if self.flashlight is not None:
            self.flashlight.on() if on else self.flashlight.off()

    def _start_blink(self):
        """Flash the head LED (4 Hz) while the alarm sounds."""
        if self.flashlight is None or self._blink_stop is not None:
            return
        stop = self._blink_stop = threading.Event()

        def run():
            on = False
            while not stop.wait(BLINK_HALF_PERIOD_S):
                on = not on
                self._flash(on)
        threading.Thread(target=run, daemon=True).start()

    def _stop_blink(self):
        if self._blink_stop is not None:
            self._blink_stop.set()
            self._blink_stop = None
            time.sleep(BLINK_HALF_PERIOD_S)        # let the blink thread finish its last toggle
            self._flash(self._lit_up and self.state == SecurityState.ARMED_ACTIVE)

    def _light_up(self):
        """Someone is here: room lights and the head flashlight on (once per spell)."""
        if self._lit_up:
            return
        self._lit_up = True
        print("[security] person seen -> room lights + flashlight on")
        self._lights(True)
        if self._blink_stop is None:
            self._flash(True)

    def _track(self, frame, faces, dets):
        """Point the robot at whoever is here: an unknown face first, then any face, then a
        person seen by YOLO (they may have turned away)."""
        if self.tracker is None:
            return
        target = None
        unknown = [f for f in faces if f.label != "owner"]
        pick = max(unknown or faces, key=lambda f: f.bbox[2], default=None)
        if pick is not None:
            target = pick.bbox[0] + pick.bbox[2] / 2
        elif dets:
            people = [d for d in dets if d.label == "person"]
            if people:
                box = max(people, key=lambda d: (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]))
                target = box.center[0]
        if target is not None:
            self.tracker.update(target, frame.shape[1])

    def _lights_down(self):
        self._lit_up = False
        self._stop_blink()
        self._flash(False)

    def _start_security_alarm(self):
        with self._alarm_lock:
            if self._security_alarm_since == float("inf"):
                return                 # already hit the cap for this incident
            if self._security_alarm_since is None:
                print("[security] ALARM")
                self._security_alarm_since = time.time()
                self._notify("🚨 ALARM sounding.", photo=True)
            self._alarm_on()

    def _security_alarm_sounding(self) -> bool:
        with self._alarm_lock:
            since = self._security_alarm_since
            return since is not None and since != float("inf")

    def _stop_security_alarm(self):
        with self._alarm_lock:
            self._security_alarm_since = None
            if time.time() >= self._voice_alarm_until and not self._manual_alarm:
                self._alarm_off()

    def _check_alarm_cap(self):
        with self._alarm_lock:
            since = self._security_alarm_since
            if since is not None and time.time() - since >= ALARM_MAX_S:
                print(f"[security] Alarm cap ({ALARM_MAX_S}s) reached - stopping")
                # Keep _security_alarm_since set so the same incident doesn't restart it.
                self._security_alarm_since = float("inf")
                if time.time() >= self._voice_alarm_until and not self._manual_alarm:
                    self._alarm_off()

    def _voice_alarm_timer(self):
        while True:
            with self._alarm_lock:
                remaining = self._voice_alarm_until - time.time()
                if remaining <= 0:
                    if ((self._security_alarm_since is None or self._security_alarm_since == float("inf"))
                            and not self._manual_alarm):
                        self._alarm_off()
                    return
            time.sleep(min(remaining, 0.5))

    def _speak_warning(self):
        self._volume.acquire()
        try:
            self.tts.speak(WARNING_TEXT)
        finally:
            self._volume.release()

    # ------------------------------------------------------------------ io helpers

    def _notify(self, text: str, photo: bool = False, frame=None):
        """Push alert; photos carry the vision overlay (boxes, keypoints, status)."""
        if self.notifier is None:
            return
        # An alert must never be able to take security mode down with it.
        try:
            if photo and frame is None:
                frame = self.camera.get_frame()
            try:
                image = overlay.annotate(frame) if frame is not None else None
            except Exception as e:
                print(f"[security] overlay failed ({e}) - sending the plain photo")
                image = frame
            self.notifier.notify(text, image)
        except Exception as e:
            print(f"[security] notification failed: {e}")

    def _start_recording(self):
        """Security footage is recorded with the vision overlay burned in."""
        return self.camera.start_recording(annotated=True)

    def _stop_recording(self, caption: str):
        """Stops a security recording and sends the clip to the owner's phone."""
        path = self.camera.stop_recording()
        if path and self.notifier is not None:
            try:
                self.notifier.notify_video(path, caption)
            except Exception as e:
                print(f"[security] video notification failed: {e}")

    def _camera_off(self):
        if self._camera_on:
            self._stop_camera()
            self._camera_on = False

    def _camera_up(self):
        if not self._camera_on:
            self._start_camera()
            self._camera_on = True

    def _lift(self, deg: int, ms: int = LIFT_MOVE_MS):
        if self.link is not None:
            self.link.lift(deg, ms)

    def _lights(self, on: bool):
        if self.hub is not None:
            self.hub.lights_on() if on else self.hub.lights_off()

    def _ensure_detectors(self):
        if self._face_recognizer is None:
            self._face_recognizer = FaceRecognizer()
        if self._person_detector is None:
            try:
                from object_detector import Detector, NcnnBackend
                self._person_detector = Detector(
                    NcnnBackend(PERSON_MODEL_DIR, PERSON_MODEL_IMGSZ), conf=PERSON_CONF)
            except Exception as e:
                print(f"[security] person detection unavailable ({e}) - faces only")
                self._person_detector = False

    # ------------------------------------------------------------------ main thread

    def _run(self, start: str = "arming"):
        if start == "remote":
            # Remote unlock while OFF: straight into verification, no exit delay.
            if self._active(remote=True) == "off":
                return
            self._idle_loop()
            return

        # --- ARMING: exit delay ---
        deadline = time.time() + EXIT_DELAY_S
        while self.state == SecurityState.ARMING and time.time() < deadline:
            if self._remote_unlock.is_set() or self._password_ok.is_set():
                break
            time.sleep(0.2)
        if self.state != SecurityState.ARMING:
            return
        if self._password_ok.is_set():
            self._disarm("password")
            return
        if not self._remote_unlock.is_set():
            self._lights(False)
            self._notify("🔒 Armed - watching for motion.")
        self.state = SecurityState.ARMED_IDLE
        self._idle_loop()

    def _idle_loop(self):
        while self.state == SecurityState.ARMED_IDLE:
            trigger = self._wait_for_trigger()
            if trigger is None:
                return
            if trigger == "password":
                self._disarm("password")
                return
            if self._active(remote=(trigger == "remote")) == "off":
                return

    def _wait_for_trigger(self):
        """ARMED_IDLE: blocks until motion, a remote unlock, or a correct code."""
        if self._pir is None:
            self._pir = get_pir_sensor()
        print("[security] ARMED-IDLE - lights off, waiting for motion...")
        while self.state == SecurityState.ARMED_IDLE:
            if self._remote_unlock.is_set():
                return "remote"
            if self._password_ok.is_set():
                return "password"
            if self._pir.wait_for_motion(timeout=0.5):
                return "motion"
        return None

    def _active(self, remote: bool) -> str:
        """ARMED_ACTIVE until disarmed ('off') or nothing seen for a while ('idle')."""
        print(f"[security] -> ARMED-ACTIVE ({'remote unlock' if remote else 'motion'})")
        self.state = SecurityState.ARMED_ACTIVE
        if not remote:
            self._lights(True)        # remote unlock: the hub already turned them on
        self._camera_up()
        self._start_voice()
        self._ensure_detectors()

        remote_deadline = None
        remote_expired = False
        lift_raised = False
        door_unlocked_by_us = False
        active_since = time.time()
        motion_photo_pending = not remote      # motion alert waits for a settled frame
        last_person_notify = 0.0

        def begin_remote():
            nonlocal remote_deadline, remote_expired, lift_raised
            self._remote_unlock.clear()
            self._countdown_expired.clear()
            remote_deadline = time.time() + REMOTE_UNLOCK_WINDOW_S
            remote_expired = False
            lift_raised = True          # on_hub_event already sent the lift up
            print(f"[security] Remote unlock: lift up, {REMOTE_UNLOCK_WINDOW_S:.0f}s to verify")
            self._notify(f"🚪 Remote unlock - verifying owner ({REMOTE_UNLOCK_WINDOW_S / 60:.0f} min "
                         f"for face or keypad code).")

        if remote:
            begin_remote()

        last_detection_at = time.time()   # any face OR a full-body person
        trespasser_active = False
        trespasser_onset_at = None
        last_warning_at = None
        last_unknown_seen_at = None
        alarm_triggered = False
        owner_seen_since = None
        owner_last_seen = 0.0
        last_person_check = 0.0
        last_person_seen_at = None
        person_recording = False          # silent recording started by a person-only detection
        unknown_since = None              # start of the current run of "unknown" sightings
        unknown_last = 0.0
        last_face_log = 0.0
        frame_interval = 1.0 / FACE_LOOP_FPS

        while self.state == SecurityState.ARMED_ACTIVE:
            loop_start = time.time()
            now = loop_start
            self._check_alarm_cap()

            if self._password_ok.is_set():
                return self._disarm("password", lift_raised=lift_raised, door_unlocked_by_us=door_unlocked_by_us)
            if self._remote_unlock.is_set():
                begin_remote()
                last_detection_at = now
            remote_mode = remote_deadline is not None

            # Remote-unlock window ran out (our timer or the keypad's) -> alarm, stay armed.
            if remote_mode and not remote_expired and (
                    now >= remote_deadline or self._countdown_expired.is_set()):
                remote_expired = True
                print("[security] Remote unlock not verified in time -> alarm")
                self._notify("⏰ Remote unlock NOT verified in time.", photo=True)
                self._start_security_alarm()

            frame = self.camera.get_frame()
            if frame is not None:
                if motion_photo_pending and now - active_since >= NOTIFY_PHOTO_DELAY_S:
                    motion_photo_pending = False
                    self._notify("👀 Motion detected - lights on, camera on.", frame=frame)
                if self._person_detector and now - last_person_check >= 1.0 / PERSON_CHECK_FPS:
                    last_person_check = now
                    dets = self._person_detector.detect(frame)
                    overlay.publish_objects(dets, source="security")
                    if any(d.label == "person" for d in dets):
                        last_person_seen_at = now
                        last_detection_at = now
                        self._light_up()
                        if now - last_person_notify >= PERSON_NOTIFY_INTERVAL_S:
                            last_person_notify = now
                            self._notify("🧍 Person detected.", frame=frame)
                results = self._face_recognizer.classify(frame)
                overlay.publish_faces(results)
                self._track(frame, results, dets if self._person_detector and now - last_person_check < 0.05 else None)
                has_owner = any(r.label == "owner" for r in results)
                now = time.time()
                if results:
                    last_detection_at = now
                    self._light_up()
                    if now - last_face_log >= FACE_LOG_INTERVAL_S:
                        last_face_log = now
                        print("[security] faces: " + ", ".join(
                            f"{r.label} {r.score:.2f} ({r.bbox[2]}px)" for r in results))

                # An unknown face counts only after UNKNOWN_CONFIRM_S of (near-)continuous
                # sightings, if big enough to judge, and not while the owner is the only face.
                big_unknown = any(r.label == "unknown" and r.bbox[2] >= MIN_UNKNOWN_FACE_PX
                                  for r in results)
                if has_owner and len(results) == 1:
                    unknown_since = None            # the owner's face flickering, not a stranger
                elif big_unknown:
                    unknown_since = unknown_since or now
                    unknown_last = now
                elif unknown_since is not None and now - unknown_last > 0.5:
                    unknown_since = None
                has_unknown = (unknown_since is not None and big_unknown
                               and now - unknown_since >= UNKNOWN_CONFIRM_S)
                if has_unknown:
                    last_unknown_seen_at = now

                # --- owner verification ---
                # During a remote unlock the owner wins even with a stranger in view.
                # Otherwise a stranger's face seen in the last few seconds blocks it
                # (the incident itself may still be on just because the person detector
                # sees someone - possibly the owner - and that mustn't lock the owner out).
                stranger_recent = (last_unknown_seen_at is not None
                                   and now - last_unknown_seen_at < UNKNOWN_ABSENCE_GRACE_S)
                if has_owner:
                    owner_last_seen = now
                    owner_seen_since = owner_seen_since or now
                elif now - owner_last_seen > OWNER_FLICKER_S:
                    owner_seen_since = None
                if (owner_seen_since is not None and now - owner_seen_since >= OWNER_CONFIRM_S
                        and (remote_mode or not stranger_recent)):
                    return self._disarm("face", remote=remote_mode, lift_raised=lift_raised,
                                        door_unlocked_by_us=door_unlocked_by_us)

                # --- trespasser response ---
                trigger_s = REMOTE_ALARM_TRIGGER_S if remote_mode else ALARM_TRIGGER_S
                if has_unknown and not trespasser_active:
                    trespasser_active = True
                    trespasser_onset_at = now
                    last_warning_at = now
                    alarm_triggered = False
                    person_recording = False   # the trespasser response owns the recording now
                    self._start_recording()   # no-op if a person-only recording is running
                    if not remote_mode and self.hub is not None and not door_unlocked_by_us:
                        print("[security] Unknown face -> unlocking door so they can leave")
                        self.hub.door_unlock()
                        door_unlocked_by_us = True
                    self._notify("⚠️ UNKNOWN PERSON - warning given, recording"
                                 + (", door unlocked so they can leave." if door_unlocked_by_us else "."),
                                 frame=frame)
                    self._speak_warning()
                elif trespasser_active:
                    # Present = unknown face OR any person (they may have turned away).
                    last_presence = max(last_unknown_seen_at, last_person_seen_at or 0.0)
                    if now - last_presence < INCIDENT_CLEAR_S:
                        if not alarm_triggered and (now - trespasser_onset_at) >= trigger_s:
                            self._start_security_alarm()
                            alarm_triggered = True
                        if (now - last_warning_at) >= TRESPASSER_WARNING_REPEAT_S:
                            self._speak_warning()
                            last_warning_at = now
                    else:
                        print(f"[security] Nobody seen for {INCIDENT_CLEAR_S:.0f}s - incident over")
                        trespasser_active = False
                        alarm_triggered = False
                        if not remote_expired:
                            self._stop_security_alarm()
                        self._stop_recording("🎥 Intruder recording.")

                # Relock the door we unlocked only once nobody has been seen for a while.
                if door_unlocked_by_us and not trespasser_active:
                    last_presence = max(last_unknown_seen_at or 0.0, last_person_seen_at or 0.0)
                    if now - last_presence >= DOOR_RELOCK_CLEAR_S:
                        print(f"[security] Nobody seen for {DOOR_RELOCK_CLEAR_S:.0f}s -> locking door again")
                        self.hub.door_lock()
                        door_unlocked_by_us = False

                # --- person without an unknown face: record silently ---
                if not trespasser_active:
                    person_recent = (last_person_seen_at is not None
                                     and now - last_person_seen_at < UNKNOWN_ABSENCE_GRACE_S)
                    if person_recent and not person_recording and not self.camera.is_recording():
                        print("[security] Person detected (no unknown face) -> recording silently")
                        self._start_recording()
                        person_recording = True
                    elif person_recording and not person_recent:
                        print("[security] Person gone -> stopping silent recording")
                        self._stop_recording("🎥 Person seen while armed.")
                        person_recording = False

                # --- nothing seen for a while -> back to motion-only ---
                in_remote_window = remote_mode and not remote_expired
                if (not trespasser_active and not in_remote_window
                        and not self._security_alarm_sounding()
                        and now - last_detection_at >= NO_DETECTION_TIMEOUT_S):
                    self._back_to_armed_idle(lift_raised, door_unlocked_by_us)
                    return "idle"

            time.sleep(max(0.0, frame_interval - (time.time() - loop_start)))
        return "off"

    def _back_to_armed_idle(self, lift_raised: bool, door_unlocked_by_us: bool):
        print(f"[security] No detection for {NO_DETECTION_TIMEOUT_S:.0f}s -> lights off, back to ARMED-IDLE")
        self._stop_security_alarm()
        if self.camera.is_recording():
            self._stop_recording("🎥 Security recording.")
        if door_unlocked_by_us:
            self.hub.door_lock()
        if lift_raised:
            self._lift(LIFT_HOME_DEG)
        self._lights_down()
        self._lights(False)
        if self.tracker is not None:
            self.tracker.release()
        self._stop_voice()
        self._camera_off()
        self.state = SecurityState.ARMED_IDLE

    def _disarm(self, method: str, remote: bool = False, lift_raised: bool = False,
                door_unlocked_by_us: bool = False) -> str:
        print(f"[security] Disarmed by {'owner face' if method == 'face' else 'keypad code'}")
        self._notify(f"✅ Disarmed by {'owner face' if method == 'face' else 'keypad code'}.")
        self.state = SecurityState.OFF
        self._stop_security_alarm()
        self._lights_down()
        if self.tracker is not None:
            self.tracker.release()
        if self.camera.is_recording():
            self._stop_recording("🎥 Security recording.")
        if method == "face" and remote and self.hub is not None:
            self.hub.cancel_countdown()   # keypad stops its countdown/buzzer
            self.hub.door_lock()
        elif door_unlocked_by_us and self.hub is not None:
            self.hub.door_lock()
        self._lift(LIFT_HOME_DEG)
        self._camera_up()
        self._start_voice()
        self._start_gestures()
        self.tts.speak("Welcome back. Security mode disarmed.")
        return "off"


if __name__ == "__main__":
    from camera_manager import CameraManager

    class _PrintTTS:
        def speak(self, text: str):
            print(f"Qwen: {text}")

    def _stub(label):
        def _fn():
            print(f"[stub] {label}")
        return _fn

    print("Testing the security-mode state machine standalone - voice/gestures are")
    print("stubbed out (just print what they'd do); face recognition, PIR, and the")
    print("alarm are real. Enroll an owner face first if you haven't:")
    print("    python face_recognizer.py --enroll\n")
    print("Type 's' + Enter to arm (simulates saying \"security mode\"). Ctrl+C to quit.\n")

    camera = CameraManager()
    camera.start()
    security = SecurityController(
        camera=camera, tts=_PrintTTS(),
        start_voice=_stub("voice pipeline started"), stop_voice=_stub("voice pipeline stopped"),
        start_gestures=_stub("gestures started"), stop_gestures=_stub("gestures stopped"),
        start_camera=camera.start, stop_camera=camera.stop,
    )

    try:
        while security.state == SecurityState.OFF:
            if input("> ").strip().lower() == "s":
                security.arm()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        security.shutdown()
        camera.stop()
