from ultralytics import YOLO
import numpy as np
from PIL import Image
from io import BytesIO
from fastapi import UploadFile
from app.models.schemas import DetectionResult, BoundingBox

model = YOLO("yolov8n.pt")

async def detect_people(file: UploadFile) -> DetectionResult:
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")
    frame = np.array(image)

    results = model(frame)
    detections = results[0].boxes.data.cpu().numpy()

    people = []
    for box in detections:
        if int(box[5]) == 0:
            x1, y1, x2, y2 = map(int, box[:4])
            people.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2))

    return DetectionResult(detections=people)
