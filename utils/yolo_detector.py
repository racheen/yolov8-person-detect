"""
YOLOv8 Person Detection Module
Handles model loading and inference for person detection
"""
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

class PersonDetector:
    """YOLOv8-based person detector"""
    
    def __init__(self, model_path="yolov8n.pt"):
        """Initialize the detector with a YOLOv8 model"""
        self.model = YOLO(model_path)
        self.person_class_id = 0  # COCO dataset person class
    
    def detect_image(self, image):
        """
        Detect people in a single image
        
        Args:
            image: numpy array or PIL Image
            
        Returns:
            tuple: (annotated_image, detection_count_text)
        """
        if image is None:
            return None, "No image provided"
        
        # Convert to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Run detection
        results = self.model(image)
        
        # Draw bounding boxes
        annotated_frame = results[0].plot()
        
        # Count person detections
        detections = results[0].boxes.data.cpu().numpy()
        person_count = sum(1 for box in detections if int(box[5]) == self.person_class_id)
        
        return annotated_frame, f"Detected {person_count} person(s)"
    
    def detect_video(self, video_path, max_frames=100):
        """
        Detect people in video frames
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to process
            
        Returns:
            numpy array: First frame with detections or None
        """
        if video_path is None:
            return None
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while cap.isOpened() and len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detection
            results = self.model(frame)
            annotated_frame = results[0].plot()
            frames.append(annotated_frame)
        
        cap.release()
        return frames[0] if frames else None
    
    def detect_frame(self, frame):
        """
        Detect people in a single frame (for webcam streaming)
        
        Args:
            frame: numpy array representing a video frame
            
        Returns:
            tuple: (annotated_frame, detection_count_text)
        """
        return self.detect_image(frame)
    
    def get_bounding_boxes(self, image):
        """
        Get raw bounding box coordinates without visualization
        
        Args:
            image: numpy array or PIL Image
            
        Returns:
            list: List of bounding boxes [(x1, y1, x2, y2, confidence), ...]
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        results = self.model(image)
        detections = results[0].boxes.data.cpu().numpy()
        
        # Filter for person class only
        person_boxes = [
            (int(box[0]), int(box[1]), int(box[2]), int(box[3]), float(box[4]))
            for box in detections if int(box[5]) == self.person_class_id
        ]
        
        return person_boxes