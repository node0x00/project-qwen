"""
Project Qwen - face recognition for security mode.

Detects all faces in a frame and classifies each independently as "owner" or
"unknown" against a single stored owner embedding. Multiple faces are
handled simultaneously and classified separately (e.g. owner home with a
guest, or several unknown people at once).

Uses OpenCV's bundled YuNet face detector + SFace face recognizer (both from
opencv_zoo, ONNX-based) rather than insightface - avoids a new, notably
heavier dependency (insightface pulls in scikit-image, easydict, etc.) for a
4GB-RAM Pi4 target. Both are already available via the installed
opencv-contrib-python (cv2.FaceDetectorYN_create / cv2.FaceRecognizerSF_create).

Run standalone to test classification against your own webcam, or to
enroll/re-enroll the owner's face:
    python face_recognizer.py            # live owner/unknown classification test
    python face_recognizer.py --enroll   # press 'e' to (re-)enroll the owner face
    python face_recognizer.py --enroll-auto   # headless: spoken prompts, averages 10 captures
"""

import argparse
import json
import os
import urllib.request
from dataclasses import dataclass

import cv2
import numpy as np

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

MODEL_DIR = "models"

# opencv_zoo's .onnx files are stored via git-lfs. The naive
# raw.githubusercontent.com/... URL pattern (used for hand_landmarker.task
# elsewhere in this project) returns a small git-lfs *pointer* text file for
# these, not the real binary - confirmed via curl (text/plain, ~130 bytes).
# The real binaries live under media.githubusercontent.com/media/... instead
# (application/octet-stream, correct byte sizes). Don't "simplify" these
# back to raw.githubusercontent.com - it'll silently download a stub.
YUNET_MODEL_PATH = os.path.join(MODEL_DIR, "face_detection_yunet_2023mar.onnx")
YUNET_MODEL_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
YUNET_MIN_BYTES = 100_000  # real file is ~227KB; a git-lfs pointer is ~130 bytes

SFACE_MODEL_PATH = os.path.join(MODEL_DIR, "face_recognition_sface_2021dec.onnx")
SFACE_MODEL_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
    "models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
)
SFACE_MIN_BYTES = 10_000_000  # real file is ~36.9MB

OWNER_EMBEDDING_PATH = os.path.join(MODEL_DIR, "owner_face.json")

# SFace's documented cosine-similarity cutoff for "same person" (FR_COSINE
# mode) - scores at or above this are treated as a match.
MATCH_THRESHOLD = 0.363
# Scores just under the match cutoff are "uncertain", not "unknown": in
# testing the owner's own face dipped to 0.22-0.36 under different lighting /
# camera height, while strangers typically score ~0-0.2. Security mode never
# treats "uncertain" as a stranger.
UNKNOWN_BELOW = 0.25

# The owner can be enrolled several times (different lighting, lift height,
# distance); a face is compared against every sample and the best match wins.
MAX_OWNER_SAMPLES = 12

# YuNet detector. opencv_zoo's own sample code defaults SCORE_THRESHOLD to
# 0.9, but that proved too strict for this webcam/lighting in practice - a
# real, well-lit, front-facing test face scored 0.868 and was being
# rejected. 0.6 leaves a wide margin below real faces (0.868 in that test)
# and above background/noise false positives (0.219 was the next-highest
# score seen on the same frame at a lower threshold). Tune against a live
# capture (see the standalone __main__ below) if this needs adjusting for
# a different camera/room.
DETECTOR_INPUT_SIZE = (320, 320)
SCORE_THRESHOLD = 0.6
NMS_THRESHOLD = 0.3
TOP_K = 5000

CAMERA_INDEX = 0


@dataclass
class FaceResult:
    bbox: tuple[int, int, int, int]  # x, y, w, h
    label: str                        # "owner" | "uncertain" | "unknown"
    score: float                      # best cosine similarity to any owner sample
    landmarks: tuple = ()             # 5 (x, y) keypoints: right eye, left eye, nose, mouth corners


# --------------------------------------------------------------------------
# Model download
# --------------------------------------------------------------------------

def _ensure_model_downloaded(path: str, url: str, min_bytes: int, label: str):
    # The size check applies even to a file that already exists, not just a
    # fresh download - otherwise a stale git-lfs pointer file left over from
    # an earlier bad run (e.g. before the media.githubusercontent.com fix)
    # would be silently trusted forever instead of being re-fetched.
    if os.path.exists(path) and os.path.getsize(path) >= min_bytes:
        return
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"Downloading {label} model (~{min_bytes // 1000}KB+) to {path}...")
    urllib.request.urlretrieve(url, path)
    size = os.path.getsize(path)
    if size < min_bytes:
        os.remove(path)
        raise RuntimeError(
            f"{path} downloaded too small ({size} bytes, expected >= {min_bytes}) - "
            f"likely got a git-lfs pointer instead of the real file. Check the URL "
            f"still points at media.githubusercontent.com, not raw.githubusercontent.com."
        )
    print("Download complete.")


def ensure_models_downloaded():
    _ensure_model_downloaded(YUNET_MODEL_PATH, YUNET_MODEL_URL, YUNET_MIN_BYTES, "YuNet detector")
    _ensure_model_downloaded(SFACE_MODEL_PATH, SFACE_MODEL_URL, SFACE_MIN_BYTES, "SFace recognizer")


# --------------------------------------------------------------------------
# Face recognizer
# --------------------------------------------------------------------------

class FaceRecognizer:
    def __init__(self):
        ensure_models_downloaded()
        self.detector = cv2.FaceDetectorYN_create(
            YUNET_MODEL_PATH, "", DETECTOR_INPUT_SIZE,
            SCORE_THRESHOLD, NMS_THRESHOLD, TOP_K,
        )
        self.recognizer = cv2.FaceRecognizerSF_create(SFACE_MODEL_PATH, "")
        self.owner_embeddings = self._load_owner_embeddings()   # list of (1, 128) arrays

    def _load_owner_embeddings(self) -> list[np.ndarray]:
        if not os.path.exists(OWNER_EMBEDDING_PATH):
            return []
        with open(OWNER_EMBEDDING_PATH) as f:
            data = json.load(f)
        # Old format: a single {"embedding": [...]}; new: {"embeddings": [[...], ...]}
        samples = data.get("embeddings") or [data["embedding"]]
        return [np.array(e, dtype=np.float32).reshape(1, -1) for e in samples]

    def save_owner_embedding(self, embedding: np.ndarray, add: bool = False):
        """Replaces the owner samples with `embedding`, or adds it to them
        (add=True, keeping the newest MAX_OWNER_SAMPLES)."""
        emb = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        self.owner_embeddings = (self.owner_embeddings + [emb])[-MAX_OWNER_SAMPLES:] if add else [emb]
        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(OWNER_EMBEDDING_PATH, "w") as f:
            json.dump({"embeddings": [e.flatten().tolist() for e in self.owner_embeddings]}, f)

    def has_owner_enrolled(self) -> bool:
        return bool(self.owner_embeddings)

    def _detect_raw(self, frame):
        """Returns the raw YuNet detections (Nx15 array: bbox + 5 landmarks +
        score per row), or None if nothing detected."""
        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame)
        return faces

    def embed(self, frame, face_row) -> np.ndarray:
        """Aligns and embeds a single detected face (one row from
        _detect_raw's output) into SFace's 128-d feature space."""
        aligned = self.recognizer.alignCrop(frame, face_row)
        return self.recognizer.feature(aligned)

    def classify(self, frame) -> list[FaceResult]:
        """Detects every face in `frame` and classifies each independently
        as owner/unknown - multi-face support is inherent since every
        detected face gets its own embedding + comparison."""
        faces = self._detect_raw(frame)
        if faces is None:
            return []

        results = []
        for row in faces:
            embedding = self.embed(frame, row)
            label, score = self._classify_embedding(embedding)
            bbox = tuple(int(v) for v in row[:4])
            landmarks = tuple((float(row[4 + 2 * i]), float(row[5 + 2 * i])) for i in range(5))
            results.append(FaceResult(bbox=bbox, label=label, score=score, landmarks=landmarks))
        return results

    def _classify_embedding(self, embedding: np.ndarray) -> tuple[str, float]:
        if not self.owner_embeddings:
            return "unknown", 0.0
        score = max(self.recognizer.match(embedding, owner, cv2.FaceRecognizerSF_FR_COSINE)
                    for owner in self.owner_embeddings)
        if score >= MATCH_THRESHOLD:
            return "owner", score
        return ("uncertain" if score >= UNKNOWN_BELOW else "unknown"), score


# --------------------------------------------------------------------------
# Standalone test / enrollment tool
# --------------------------------------------------------------------------

def _draw_results(frame, results: list[FaceResult]):
    for r in results:
        x, y, w, h = r.bbox
        color = {"owner": (0, 200, 0), "uncertain": (0, 200, 255)}.get(r.label, (0, 0, 255))
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{r.label} ({r.score:.2f})", (x, max(0, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


if __name__ == "__main__":
    from camera_manager import CameraManager

    parser = argparse.ArgumentParser(description="Face recognizer standalone test/enroll tool.")
    parser.add_argument("--enroll", action="store_true",
                         help="Enrollment mode - press 'e' to (re-)enroll the owner's face.")
    parser.add_argument("--enroll-auto", action="store_true",
                         help="Headless enrollment (no window): spoken prompts, averages "
                              "several captures of exactly one face.")
    parser.add_argument("--add", action="store_true",
                         help="With --enroll-auto: add this sample to the owner's samples "
                              "instead of replacing them (enroll several conditions).")
    parser.add_argument("--lift", type=int, default=None,
                         help="With --enroll-auto: move the robot's lift to this angle first "
                              "(e.g. 130 = remote-unlock verification height).")
    parser.add_argument("--lights", choices=("on", "off"), default=None,
                         help="With --enroll-auto: switch the room lights through the "
                              "smart-home hub first.")
    args = parser.parse_args()

    face_recognizer = FaceRecognizer()

    if args.enroll_auto:
        import time
        import numpy as np
        from assistant import TTS

        tts = TTS()
        samples_needed, sample_gap_s, give_up_s = 10, 0.4, 60
        if args.lights is not None:
            import urllib.request
            from smart_home import HUB_URL
            urllib.request.urlopen(f"{HUB_URL}/lights/{args.lights}", timeout=5).read()
            print(f"room lights {args.lights}")
            time.sleep(2)            # light-switch servos
        if args.lift is not None:
            from esp32_link import ESP32Link
            lift_link = ESP32Link()
            lift_link.start()
            lift_link.lift(args.lift, 1500)
            lift_link.wait_lift(5)
            print(f"lift at {lift_link.state.lift_pos}")
            lift_link.close()
        camera = CameraManager(camera_index=CAMERA_INDEX)
        camera.start()
        try:
            time.sleep(1.5)   # auto-exposure settle
            tts.speak("Face enrollment. Look at my camera, alone, and hold still.")
            samples, last_sample, last_hint, start = [], 0.0, 0.0, time.time()
            while len(samples) < samples_needed and time.time() - start < give_up_s:
                frame = camera.get_frame()
                if frame is None:
                    time.sleep(0.05)
                    continue
                faces = face_recognizer._detect_raw(frame)
                n = 0 if faces is None else len(faces)
                now = time.time()
                if n == 1 and now - last_sample >= sample_gap_s:
                    samples.append(face_recognizer.embed(frame, faces[0]).flatten())
                    last_sample = now
                    print(f"captured {len(samples)}/{samples_needed}")
                elif n != 1 and now - last_hint > 8:
                    print(f"need exactly 1 face in frame (see {n})")
                    tts.speak("I can't see your face." if n == 0 else "I see more than one face.")
                    last_hint = now
                time.sleep(0.05)
            if len(samples) < samples_needed:
                tts.speak("Enrollment failed, I couldn't see your face clearly.")
                raise SystemExit("enrollment failed")
            # Average the unit-normalized embeddings, then renormalize.
            unit = [e / np.linalg.norm(e) for e in samples]
            mean = np.mean(unit, axis=0)
            face_recognizer.save_owner_embedding(mean / np.linalg.norm(mean), add=args.add)
            print(f"Owner face enrolled -> {OWNER_EMBEDDING_PATH} "
                  f"({len(face_recognizer.owner_embeddings)} sample(s))")
            tts.speak("Got it. I'll recognize you now.")
        finally:
            camera.stop()
        raise SystemExit(0)

    if args.enroll:
        print("Enrollment mode. Look at the camera alone (exactly one face in frame),")
        print("then press 'e' to capture, or 'q' to quit without enrolling.")
    else:
        print("Testing face classification. Green box = owner, red box = unknown.")
        print("Press 'q' to quit.")
        if not face_recognizer.has_owner_enrolled():
            print("(No owner enrolled yet - everyone will show as 'unknown'. "
                  "Run with --enroll first.)")

    camera = CameraManager(camera_index=CAMERA_INDEX)
    camera.start()
    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                continue

            faces = face_recognizer._detect_raw(frame)
            display = frame.copy()

            if args.enroll:
                n = 0 if faces is None else len(faces)
                if faces is not None:
                    for row in faces:
                        x, y, w, h = (int(v) for v in row[:4])
                        cv2.rectangle(display, (x, y), (x + w, y + h), (255, 200, 0), 2)
                status = "press 'e' to enroll" if n == 1 else f"need exactly 1 face (see {n})"
                cv2.putText(display, status, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 200, 0), 2)
            else:
                results = face_recognizer.classify(frame)
                _draw_results(display, results)

            display = cv2.flip(display, 1)
            cv2.imshow("Face recognizer", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if args.enroll and key == ord('e'):
                if faces is not None and len(faces) == 1:
                    embedding = face_recognizer.embed(frame, faces[0])
                    face_recognizer.save_owner_embedding(embedding)
                    print(f"Owner face enrolled -> {OWNER_EMBEDDING_PATH}")
                else:
                    n = 0 if faces is None else len(faces)
                    print(f"Need exactly 1 face in frame to enroll (saw {n}) - try again.")
    finally:
        cv2.destroyAllWindows()
        camera.stop()
