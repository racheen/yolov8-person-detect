---
title: YOLOv8 Person Detection
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
python_version: "3.11"
app_file: app.py
pinned: false
---

# YOLOv8 Person Detection

Real-time person detection using [Ultralytics YOLOv8](https://docs.ultralytics.com/). Upload images, videos, or use your webcam!

---

## Features

- **Image Upload**: Detect people in static images
- **Webcam Stream**: Real-time detection from your webcam
- **Video Upload**: Process video files frame by frame
- YOLOv8n model optimized for person detection
- Shareable demo link


## Project Structure
```
.
├── main.py
├── core/
│   └── config.py
├── api/
│   └── routes/
│       ├── detect.py
│       └── websocket.py
├── models/
│   └── schemas.py
└── services/
    └── yolo_service.py
```


## Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd yolov8-person-detect
```

2. **Install dependencies**
```
pip install -r requirements.txt
```

3. **Run the app**
```
python app.py
```
The app will launch locally and provide a public shareable link (valid for 72 hours).



## Usage

### Image Detection
1. Go to "Image Upload" tab
2. Upload an image (JPEG/PNG)
3. Click "Detect People"
4. View annotated results with bounding boxes

### Webcam Detection
1. Go to "Webcam (Real-time)" tab
2. Allow camera access
3. See live detection results

### Video Detection
1. Go to "Video Upload" tab
2. Upload a video file
3. Click "Process Video"
4. View detection on first frame


## Model Details

- **Model**: YOLOv8n (nano) - fastest variant
- **Detection**: Person class only (COCO class 0)
- **Output**: Bounding boxes with confidence scores