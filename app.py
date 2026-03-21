import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Load model
model = YOLO("yolov8n.pt")

def detect_people_image(image):
    """Detect people in a single image"""
    if image is None:
        return None
    
    # Convert to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Run detection
    results = model(image)
    
    # Draw bounding boxes
    annotated_frame = results[0].plot()
    
    # Count detections
    detections = results[0].boxes.data.cpu().numpy()
    person_count = sum(1 for box in detections if int(box[5]) == 0)
    
    return annotated_frame, f"Detected {person_count} person(s)"

def detect_people_video(video):
    """Detect people in video frames"""
    cap = cv2.VideoCapture(video)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run detection
        results = model(frame)
        annotated_frame = results[0].plot()
        frames.append(annotated_frame)
        
        # Limit to first 100 frames for demo
        if len(frames) >= 100:
            break
    
    cap.release()
    return frames[0] if frames else None

def detect_people_webcam(image):
    """Detect people from webcam stream"""
    return detect_people_image(image)

# Create Gradio interface with tabs
with gr.Blocks(title="YOLOv8 Person Detection") as demo:
    gr.Markdown("# 🧠 YOLOv8 Person Detection")
    gr.Markdown("Upload an image, video, or use your webcam to detect people in real-time")
    
    with gr.Tab("Image Upload"):
        with gr.Row():
            image_input = gr.Image(type="numpy", label="Upload Image")
            image_output = gr.Image(label="Detection Result")
        image_text = gr.Textbox(label="Detection Count")
        image_button = gr.Button("Detect People")
        image_button.click(
            fn=detect_people_image,
            inputs=image_input,
            outputs=[image_output, image_text]
        )
    
    with gr.Tab("Webcam (Real-time)"):
        gr.Markdown("⚠️ Allow camera access when prompted")
        webcam_input = gr.Image(source="webcam", streaming=True, label="Webcam Feed")
        webcam_output = gr.Image(label="Detection Result")
        webcam_text = gr.Textbox(label="Detection Count")
        
        webcam_input.stream(
            fn=detect_people_webcam,
            inputs=webcam_input,
            outputs=[webcam_output, webcam_text]
        )
    
    with gr.Tab("Video Upload"):
        with gr.Row():
            video_input = gr.Video(label="Upload Video")
            video_output = gr.Image(label="First Frame Detection")
        video_button = gr.Button("Process Video")
        video_button.click(
            fn=detect_people_video,
            inputs=video_input,
            outputs=video_output
        )
    
    gr.Markdown("### 📝 About")
    gr.Markdown("This demo uses YOLOv8n model to detect people in images and videos.")

if __name__ == "__main__":
    demo.launch(share=True)  # share=True creates a public link