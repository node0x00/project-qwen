"""
Project Qwen - on-demand object / person detection (YOLO26n, COCO).

Runs only when asked ("what do you see", "is anyone here", security mode's
throttled person check) - never continuously - to save CPU and battery.

Two interchangeable backends behind one interface, so calling code never
changes when the model format does:
  - NcnnBackend        the Pi: plain `ncnn` runtime + NCNN export, no PyTorch
  - UltralyticsBackend the dev Mac: Ultralytics + PyTorch weights (.pt)
get_detector() picks whichever is available (NCNN first).

The NCNN model is exported on the Mac by vision-tools/export_yolo.py. Its
output `out0` is (84, anchors): 4 box values (center x, center y, width,
height in letterboxed input pixels - verified against Ultralytics' own
predictions) + 80 class scores (already sigmoid). YOLO26's one-to-one head is NMS-free, so
decoding is "best class per anchor, keep the confident ones"; a light NMS is
still applied as a safety net against the odd duplicate.

Standalone test (one frame from the camera, prints detections + timing):
    python object_detector.py [image.jpg]
"""

import os
import time
from dataclasses import dataclass

import cv2
import numpy as np

MODEL_DIR = "models"
NCNN_MODEL = os.path.join(MODEL_DIR, "yolo26n_640_ncnn_model")
NCNN_IMGSZ = 640
PT_WEIGHTS = "yolo26n.pt"

CONF_THRESHOLD = 0.35
NMS_IOU = 0.6
NCNN_THREADS = 4

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]


@dataclass
class Detection:
    label: str
    confidence: float
    box: tuple[float, float, float, float]   # x1, y1, x2, y2 in frame pixels

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2, (y1 + y2) / 2

    def cut_off_edges(self, frame_shape, margin: int = 4) -> set[str]:
        """Which frame edges the box touches ('left', 'right', 'top', 'bottom')
        - a box touching an edge is probably cut off there."""
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = self.box
        edges = set()
        if x1 <= margin:
            edges.add("left")
        if x2 >= w - 1 - margin:
            edges.add("right")
        if y1 <= margin:
            edges.add("top")
        if y2 >= h - 1 - margin:
            edges.add("bottom")
        return edges


# ---------------------------------------------------------------------- backends

def _letterbox(frame, size):
    """Resize keeping aspect ratio, pad to size x size with gray (Ultralytics'
    own preprocessing). Returns the padded image, scale, and (pad_x, pad_y)."""
    h, w = frame.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = round(h * scale), round(w * scale)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    out = np.full((size, size, 3), 114, np.uint8)
    out[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return out, scale, (pad_x, pad_y)


class NcnnBackend:
    name = "ncnn"

    def __init__(self, model_dir: str = NCNN_MODEL, imgsz: int = NCNN_IMGSZ,
                 threads: int = NCNN_THREADS):
        import ncnn
        self._ncnn = ncnn
        self.imgsz = imgsz
        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = False
        self.net.opt.num_threads = threads
        if self.net.load_param(os.path.join(model_dir, "model.ncnn.param")) != 0 or \
                self.net.load_model(os.path.join(model_dir, "model.ncnn.bin")) != 0:
            raise RuntimeError(f"couldn't load NCNN model from {model_dir}")

    def detect(self, frame, conf: float) -> list[Detection]:
        img, scale, (pad_x, pad_y) = _letterbox(frame, self.imgsz)
        chw = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32) / 255.0
        ex = self.net.create_extractor()
        ex.input("in0", self._ncnn.Mat(chw))
        _, out = ex.extract("out0")
        pred = np.array(out)                       # (84, anchors)
        cxcywh, scores = pred[:4].T, pred[4:].T
        boxes = np.column_stack([cxcywh[:, 0] - cxcywh[:, 2] / 2, cxcywh[:, 1] - cxcywh[:, 3] / 2,
                                 cxcywh[:, 0] + cxcywh[:, 2] / 2, cxcywh[:, 1] + cxcywh[:, 3] / 2])
        cls = scores.argmax(1)
        confs = scores[np.arange(len(cls)), cls]
        keep = confs >= conf
        boxes, cls, confs = boxes[keep], cls[keep], confs[keep]
        if len(confs) == 0:
            return []
        # Undo letterbox, clip to the frame.
        boxes = (boxes - [pad_x, pad_y, pad_x, pad_y]) / scale
        h, w = frame.shape[:2]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h - 1)
        xywh = np.column_stack([boxes[:, 0], boxes[:, 1], boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]])
        idx = cv2.dnn.NMSBoxesBatched(xywh.tolist(), confs.tolist(), cls.tolist(), conf, NMS_IOU)
        return [Detection(COCO_NAMES[cls[i]], float(confs[i]), tuple(float(v) for v in boxes[i]))
                for i in np.array(idx).flatten()]


class UltralyticsBackend:
    name = "ultralytics"

    def __init__(self, weights: str = PT_WEIGHTS, imgsz: int = NCNN_IMGSZ):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.imgsz = imgsz

    def detect(self, frame, conf: float) -> list[Detection]:
        res = self.model.predict(frame, imgsz=self.imgsz, conf=conf, verbose=False)[0]
        return [Detection(res.names[int(c)], float(p), tuple(float(v) for v in b))
                for b, c, p in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(),
                                   res.boxes.conf.tolist())]


# ---------------------------------------------------------------------- public API

class Detector:
    def __init__(self, backend, conf: float = CONF_THRESHOLD):
        self.backend = backend
        self.conf = conf

    def detect(self, frame, conf: float | None = None) -> list[Detection]:
        return self.backend.detect(frame, self.conf if conf is None else conf)


def get_detector(conf: float = CONF_THRESHOLD) -> Detector:
    """NCNN if the runtime and exported model are present (the Pi), otherwise
    Ultralytics + PyTorch weights (the Mac)."""
    errors = []
    for factory in (NcnnBackend, UltralyticsBackend):
        try:
            backend = factory()
            print(f"[detector] using {backend.name} backend")
            return Detector(backend, conf)
        except Exception as e:
            errors.append(f"{factory.name}: {e}")
    raise RuntimeError("no object detection backend available - " + "; ".join(errors))


def _plural(label: str, n: int) -> str:
    if n == 1:
        return label
    irregular = {"person": "people", "mouse": "mice", "knife": "knives", "skis": "skis",
                 "sheep": "sheep", "scissors": "scissors", "couch": "couches",
                 "bus": "buses", "toothbrush": "toothbrushes", "sandwich": "sandwiches",
                 "wine glass": "wine glasses", "glass": "glasses", "hair drier": "hair driers",
                 "sports ball": "sports balls", "potted plant": "potted plants"}
    return irregular.get(label, label + "s")


_NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
_ARTICLE_OVERRIDES = {"tv": "a", "hair drier": "a"}


def describe(detections: list[Detection]) -> str:
    """Natural spoken summary: 'I see two people, a laptop and a chair.'"""
    if not detections:
        return "I don't see anything I recognize."
    counts: dict[str, int] = {}
    for d in sorted(detections, key=lambda d: -d.confidence):
        counts[d.label] = counts.get(d.label, 0) + 1
    # People first, then by count.
    items = sorted(counts.items(), key=lambda kv: (kv[0] != "person", -kv[1]))
    parts = []
    for label, n in items:
        if n == 1:
            article = _ARTICLE_OVERRIDES.get(label, "an" if label[0] in "aeiou" else "a")
            parts.append(f"{article} {label}")
        else:
            parts.append(f"{_NUMBER_WORDS[n] if n < len(_NUMBER_WORDS) else n} {_plural(label, n)}")
    if len(parts) == 1:
        return f"I see {parts[0]}."
    return f"I see {', '.join(parts[:-1])} and {parts[-1]}."


if __name__ == "__main__":
    import sys

    detector = get_detector()
    if len(sys.argv) > 1:
        frame = cv2.imread(sys.argv[1])
    else:
        cap = cv2.VideoCapture(0)
        for _ in range(10):       # let auto-exposure settle
            ok, frame = cap.read()
        cap.release()
    detector.detect(frame)        # warm-up run
    t = time.time()
    dets = detector.detect(frame)
    print(f"{len(dets)} detections in {(time.time() - t) * 1000:.0f} ms")
    for d in dets:
        print(f"  {d.label:14} {d.confidence:.2f}  box={tuple(round(v) for v in d.box)}  "
              f"edges={d.cut_off_edges(frame.shape) or '-'}")
    print(describe(dets))
