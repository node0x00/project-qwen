"""
Project Qwen - camera hand-off between the assistant's vision and the dock-tag tracker.

There's one C920. Normally CameraManager owns it (voice/gesture/security vision and the live
view). While the robot docks or drives itself (RobotController auto tasks), mono_live runs in
tag-only mode instead: it writes the tag pose (/dev/shm/slam_tag.txt), a clean frame for the
docked-view matching (/dev/shm/cam.jpg) and the annotated control view (VIEW).

acquire() hands the camera to mono_live (pausing gestures first); release() gives it back after
RELEASE_DELAY_S, so back-to-back docking steps don't bounce the camera. Whatever the camera's
state was before (security mode may have it off), it's put back the same way.
"""

import os
import subprocess
import threading
import time

SLAM_DIR = os.path.expanduser("~/slam")
VIEW = "/dev/shm/slam_view.jpg"
TAG_FILE = "/dev/shm/slam_tag.txt"
RELEASE_DELAY_S = 20.0
START_TIMEOUT_S = 15.0


class DockCamera:
    def __init__(self, camera, pause_vision=None, resume_vision=None):
        """camera: CameraManager. pause_vision / resume_vision: callables (gestures etc.)."""
        self.camera = camera
        self.pause_vision = pause_vision or (lambda: None)
        self.resume_vision = resume_vision or (lambda: None)
        self._lock = threading.Lock()
        self._users = 0
        self._proc = None
        self._release_timer = None
        self.camera_wanted = True       # what CameraManager should be when we hand back
        self.slam = False               # full ORB-SLAM (lost-tag search) instead of tag-only

    @property
    def active(self) -> bool:
        return self._proc is not None

    def acquire(self):
        with self._lock:
            self._users += 1
            if self._release_timer is not None:
                self._release_timer.cancel()
                self._release_timer = None
            if self._proc is not None:
                return
            print("[dockcam] taking the camera for the tag tracker", flush=True)
            self.pause_vision()
            self.camera_wanted = bool(getattr(self.camera, "_running", False))
            if self.camera_wanted:
                self.camera.stop()
            for f in (VIEW + ".txt",):
                try:
                    open(f, "w").close()
                except OSError:
                    pass
            self._proc = subprocess.Popen(
                [f"{SLAM_DIR}/qwen_slam/build/mono_live", f"{SLAM_DIR}/ORB_SLAM3/Vocabulary/ORBvoc.txt",
                 f"{SLAM_DIR}/qwen_slam/c920e.yaml", "100000", VIEW] + ([] if self.slam else ["tagonly"]),
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        end = time.time() + START_TIMEOUT_S          # wait for the first tag-file write
        while time.time() < end:
            try:
                if time.time() - os.stat(TAG_FILE).st_mtime < 1.0:
                    return
            except OSError:
                pass
            time.sleep(0.2)
        print("[dockcam] tag tracker slow to start", flush=True)

    def release(self, delay: float = RELEASE_DELAY_S):
        with self._lock:
            self._users = max(0, self._users - 1)
            if self._users or self._proc is None:
                return
            self._release_timer = threading.Timer(delay, self._hand_back)
            self._release_timer.daemon = True
            self._release_timer.start()

    def _hand_back(self):
        with self._lock:
            if self._users or self._proc is None:
                return
            proc, self._proc, self._release_timer = self._proc, None, None
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            print("[dockcam] camera back to the assistant", flush=True)
            if self.camera_wanted:
                try:
                    self.camera.start()
                except RuntimeError as e:
                    print(f"[dockcam] camera restart failed: {e}", flush=True)
            self.resume_vision()

    def set_slam(self, on: bool):
        """Restart the tracker with ORB-SLAM on (lost-tag search) or off (tag-only docking).
        The tag detection keeps running either way. Call while the camera is acquired."""
        with self._lock:
            if self.slam == on:
                return
            self.slam = on
            proc, self._proc = self._proc, None
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        with self._lock:
            self._proc = subprocess.Popen(
                [f"{SLAM_DIR}/qwen_slam/build/mono_live", f"{SLAM_DIR}/ORB_SLAM3/Vocabulary/ORBvoc.txt",
                 f"{SLAM_DIR}/qwen_slam/c920e.yaml", "100000", VIEW] + ([] if on else ["tagonly"]),
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"[dockcam] tracker restarted {'with ORB-SLAM' if on else 'tag-only'}", flush=True)

    # security mode turns the camera on/off; while docking just remember what it wants
    def start_camera(self):
        with self._lock:
            if self._proc is not None:
                self.camera_wanted = True
                return
        self.camera.start()

    def stop_camera(self):
        with self._lock:
            if self._proc is not None:
                self.camera_wanted = False
                return
        self.camera.stop()

    def shutdown(self):
        with self._lock:
            self._users = 0
            if self._release_timer is not None:
                self._release_timer.cancel()
            proc, self._proc = self._proc, None
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
