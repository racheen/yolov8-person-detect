# YOLOv8 FastAPI Backend

A FastAPI-based backend service for real-time person detection using the [Ultralytics YOLOv8](https://docs.ultralytics.com/) model. This backend receives images, runs object detection (focused on class `person`), and returns bounding box coordinates.

---

## Features

- YOLOv8-powered object detection (person class only)
- Clean modular FastAPI architecture
- CORS-enabled for frontend communication
- Ready for deployment on Render or any cloud provider

---

## Project Structure
```
.
├── README.md
├── app
│   ├── core
│   ├── main.py
│   ├── models
│   │   └── schema.py
│   ├── routes
│   │   └── api
│   │       └── detect.py
│   └── services
│       └── yolo_service.py
└── requirements.txt
```

---

## API Endpoint

### `POST /api/detect`

Uploads an image (as form-data) and returns detected people with bounding box coordinates.

#### 🔧 Request
- `Content-Type: multipart/form-data`
- `file`: image file (JPEG/PNG)

#### Response

```json
{
  "detections": [
    {
      "x1": 123,
      "y1": 45,
      "x2": 300,
      "y2": 400
    },
  ]
}
```