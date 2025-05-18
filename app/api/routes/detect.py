from fastapi import APIRouter, UploadFile, File
from app.services.yolo_service import detect_people
from app.models.schemas import DetectionResult

router = APIRouter()

@router.post("/detect", response_model=DetectionResult)
async def detect(file: UploadFile = File(...)):
    return await detect_people(file)