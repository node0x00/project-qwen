"""
Project Qwen - computer-vision debug overlay.

Every vision component publishes what it just computed:
  hands      gesture_recognizer (21 landmarks + skeleton + gesture label)
  faces      security_mode / the stream's observer (box, 5 keypoints, owner/uncertain/unknown + score)
  objects    YOLO (scene_actions, security_mode person check, observer)
  tags       AprilTags (docking - later)
and annotate(frame) draws everything still fresh on a copy of the frame, plus
a status bar (security state, REC, ALARM, lift, IR).

Used by:
  - stream_server.py   live debug view in a browser / VS Code Simple Browser
  - security_mode.py   Telegram photos and security recordings are annotated

A single module-level `overlay` instance is shared by everything (one robot,
one camera), so components just `from vision_overlay import overlay`.
"""

import threading
import time

import cv2

# How long a published result stays on screen. Hands/faces update several
# times a second while their detector runs; YOLO is slower.
STALE_S = {"hands": 0.6, "faces": 0.8, "objects": 1.5, "tags": 0.8}

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]
FINGERTIPS = (4, 8, 12, 16, 20)

# BGR
GREEN, RED, YELLOW, CYAN, WHITE, ORANGE, MAGENTA, BLACK = (
    (0, 200, 0), (0, 0, 255), (0, 220, 255), (255, 200, 0), (255, 255, 255),
    (0, 140, 255), (255, 0, 255), (0, 0, 0))
FACE_COLORS = {"owner": GREEN, "uncertain": YELLOW, "unknown": RED, "face": CYAN}
FACE_POINT_COLORS = (CYAN, CYAN, MAGENTA, ORANGE, ORANGE)   # eyes, nose, mouth corners
FONT = cv2.FONT_HERSHEY_SIMPLEX


class VisionOverlay:
    def __init__(self):
        self._lock = threading.Lock()
        self._items: dict[str, tuple[float, object]] = {}
        self._publish_times: dict[str, list[float]] = {}   # recent publish times per kind (fps readout)
        self.status_fn = None        # callable() -> list[str], set by assistant.py

    # ------------------------------------------------------------------ publishing

    def publish_hands(self, hands):
        """hands: list of (landmarks, label) - landmarks are MediaPipe
        normalized (x, y in 0..1) points, label e.g. 'thumbs_up'."""
        self._put("hands", hands)

    def publish_faces(self, faces):
        """faces: list of objects with .bbox (x, y, w, h), .landmarks (5 (x, y)
        points, may be empty), .label, .score."""
        self._put("faces", faces)

    def publish_objects(self, detections, source: str = ""):
        """detections: object_detector.Detection list (label, confidence, box xyxy)."""
        self._put("objects", (source, detections))

    def publish_tags(self, tags):
        """tags: list of objects with .corners (4 (x, y)), .tag_id, optional .text."""
        self._put("tags", tags)

    def _put(self, kind, payload):
        now = time.time()
        with self._lock:
            self._items[kind] = (now, payload)
            times = self._publish_times.setdefault(kind, [])
            times.append(now)
            del times[:-20]

    def rates(self) -> dict[str, float]:
        """Publishes per second over the last ~2 s, per kind (0 if idle)."""
        now = time.time()
        out = {}
        with self._lock:
            for kind, times in self._publish_times.items():
                recent = [t for t in times if now - t <= 2.0]
                out[kind] = (len(recent) - 1) / (recent[-1] - recent[0]) if len(recent) >= 2 and recent[-1] > recent[0] else 0.0
        return out

    def hand_visible(self, max_age: float = 1.0) -> bool:
        """True if hand tracking saw a hand within max_age seconds."""
        with self._lock:
            item = self._items.get("hands")
        return item is not None and time.time() - item[0] <= max_age and bool(item[1])

    def fresh(self, kind: str, max_age: float | None = None) -> bool:
        with self._lock:
            item = self._items.get(kind)
        return item is not None and time.time() - item[0] <= (max_age or STALE_S[kind])

    def _fresh_items(self):
        now = time.time()
        with self._lock:
            return {k: p for k, (t, p) in self._items.items() if now - t <= STALE_S.get(k, 1.0)}

    # ------------------------------------------------------------------ drawing

    def annotate(self, frame):
        """Returns an annotated copy of `frame` (the original is untouched)."""
        if frame is None:
            return None
        out = frame.copy()
        items = self._fresh_items()
        if "objects" in items:
            self._draw_objects(out, *items["objects"])
        if "faces" in items:
            self._draw_faces(out, items["faces"])
        if "hands" in items:
            self._draw_hands(out, items["hands"])
        if "tags" in items:
            self._draw_tags(out, items["tags"])
        self._draw_status(out)
        return out

    @staticmethod
    def _label(img, text, x, y, color):
        (tw, th), _ = cv2.getTextSize(text, FONT, 0.5, 1)
        y = max(th + 4, y)
        cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y + 2), color, -1)
        cv2.putText(img, text, (x + 2, y - 2), FONT, 0.5, BLACK, 1, cv2.LINE_AA)

    def _draw_objects(self, img, source, detections):
        for d in detections:
            x1, y1, x2, y2 = (int(v) for v in d.box)
            color = ORANGE if d.label == "person" else WHITE
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            self._label(img, f"{d.label} {d.confidence:.2f}", x1, y1, color)

    def _draw_faces(self, img, faces):
        for f in faces:
            x, y, w, h = (int(v) for v in f.bbox)
            color = FACE_COLORS.get(f.label, CYAN)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            for (px, py), pc in zip(getattr(f, "landmarks", ()) or (), FACE_POINT_COLORS):
                cv2.circle(img, (int(px), int(py)), 3, pc, -1)
            text = f.label if f.score is None else f"{f.label} {f.score:.2f}"
            self._label(img, text, x, y + h + 18, color)

    @staticmethod
    def _draw_hands(img, hands):
        h, w = img.shape[:2]
        for landmarks, label in hands:
            pts = [(int(p.x * w), int(p.y * h)) for p in landmarks]
            for a, b in HAND_CONNECTIONS:
                cv2.line(img, pts[a], pts[b], WHITE, 2, cv2.LINE_AA)
            for i, p in enumerate(pts):
                cv2.circle(img, p, 4, RED if i == 4 else YELLOW if i in FINGERTIPS else GREEN, -1)
            if label and label != "unknown":
                x0, y0 = min(p[0] for p in pts), min(p[1] for p in pts)
                VisionOverlay._label(img, label, x0, y0 - 6, GREEN)

    def _draw_tags(self, img, tags):
        for t in tags:
            pts = [(int(x), int(y)) for x, y in t.corners]
            for i in range(4):
                cv2.line(img, pts[i], pts[(i + 1) % 4], MAGENTA, 2)
            cx = sum(p[0] for p in pts) // 4
            cy = sum(p[1] for p in pts) // 4
            self._label(img, getattr(t, "text", None) or f"tag {t.tag_id}", cx, cy, MAGENTA)

    def _draw_status(self, img):
        parts = []
        if self.status_fn is not None:
            try:
                parts = list(self.status_fn())
            except Exception:
                parts = []
        rates = self.rates()
        fps = " · ".join(f"{name} {rates[key]:.0f}" for key, name in
                         (("hands", "hands"), ("faces", "faces"), ("objects", "yolo"))
                         if rates.get(key, 0) >= 0.5)
        if fps:
            parts.append(fps + " fps")
        parts.append(time.strftime("%H:%M:%S"))
        text = "  |  ".join(parts)
        h, w = img.shape[:2]
        cv2.rectangle(img, (0, h - 24), (w, h), BLACK, -1)
        cv2.putText(img, text, (6, h - 7), FONT, 0.5, WHITE, 1, cv2.LINE_AA)


overlay = VisionOverlay()
