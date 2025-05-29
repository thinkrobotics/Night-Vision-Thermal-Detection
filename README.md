
# Thermal Night Vision YOLO Detection System

This project implements a thermal imaging object detection system using YOLO models (via the Ultralytics library). It supports real-time detection from thermal and RGB video sources including webcams, video files, and RTSP streams.

---

## 🔧 Features

- YOLO-based object detection for thermal imagery
- Pseudo-coloring for grayscale thermal images
- Real-time detection and annotation
- Class-specific color coding:
  - Person: Green
  - Vehicle: Blue
  - Animal: Red
  - Drone: Cyan
  - Other: Magenta
- Save annotated output video (optional)
- FPS overlay on live feed
- Timestamped console logging of detections

---

## 🗂️ Project Structure

```
thermal_yolo_detector/
├── thermal_yolo_detector.py   # Main script
├── yolov8_model.pt            # Your YOLO model weights
└── README.md                  # Project documentation
```

---

## 🖥️ Requirements

- Python 3.8+
- OpenCV
- NumPy
- Ultralytics (YOLO)

Install dependencies:

```bash
pip install opencv-python numpy ultralytics
```

---

## 🚀 Usage

### Run detection:

```bash
python thermal_yolo_detector.py --model yolov8_model.pt --source 0 --thermal
```

### Additional arguments:

- `--model`: Path to YOLOv8 model weights (e.g., yolov8n.pt)
- `--source`: Source (0 = webcam, path to video file, or RTSP URL)
- `--threshold`: Confidence threshold (default=0.5)
- `--thermal`: Flag to indicate input is a thermal stream
- `--save`: Path to save annotated output video

---

## 💡 Example Commands

```bash
# Thermal camera from webcam
python thermal_yolo_detector.py --model yolov8n.pt --source 0 --thermal

# Video file input
python thermal_yolo_detector.py --model yolov8n.pt --source test_video.mp4

# Save annotated video
python thermal_yolo_detector.py --model yolov8n.pt --source 0 --thermal --save output.avi
```

---

## 📸 Output

- Annotated frames displayed live with bounding boxes and FPS
- Console prints:
  ```
  [2025-05-29 12:34:56] Detected Person with confidence 0.92 at (x1, y1, x2, y2)
  ```

---

## ⚠️ Notes

- Make sure your thermal camera is accessible by OpenCV
- For some RTSP cameras, install `ffmpeg` for compatibility:
  ```bash
  pip install imageio[ffmpeg]
  ```

---

## 📄 License

This project is under MIT License.
