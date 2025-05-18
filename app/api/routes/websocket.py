from fastapi import APIRouter, WebSocket
from app.services.yolo_service import detect_from_frame
from app.models.schemas import DetectionResult
import base64
import numpy as np
import cv2

router = APIRouter()

@router.websocket("/detect")
async def detect_websocket(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            image_data = data.split(",")[1]  # remove data:image/jpeg;base64,
            image_bytes = base64.b64decode(image_data)
            np_arr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            detections = detect_from_frame(frame)
            result = DetectionResult(detections=detections)
            await websocket.send_text(result.json())
    except Exception as e:
        await websocket.close()
