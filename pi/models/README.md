# Models

Only `hey_qwen.onnx` (the custom "Hey Qwen" wake word, trained with openWakeWord) is in the repo.
Download the others into this folder:

| File | What it is | Where to get it |
|---|---|---|
| `en_US-lessac-medium.onnx` + `.onnx.json` | Piper text-to-speech voice | [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/lessac/medium) |
| `hand_landmarker.task` | MediaPipe hand landmarks (gestures) | downloaded automatically by `gesture_recognizer.py` on first run |
| `face_detection_yunet_2023mar.onnx` | YuNet face detector | [opencv_zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) |
| `face_recognition_sface_2021dec.onnx` | SFace face embeddings | [opencv_zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) |
| `yolo26n_320_ncnn_model/`, `yolo26n_640_ncnn_model/` | YOLO26n person/object detection, NCNN export | export with Ultralytics: `yolo export model=yolo26n.pt format=ncnn imgsz=320` (and 640) |

The owner's face is enrolled on the robot with `python face_recognizer.py --enroll-auto`
and stored as `owner_face.json` here; it is ignored by git and never leaves the robot.

The alarm sound (`pi/sounds/alarm.wav`) isn't included either: drop any alarm/siren WAV there.
