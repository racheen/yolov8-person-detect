from pydantic import BaseModel
from typing import List

class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int

class DetectionResult(BaseModel):
    detections: List[BoundingBox]