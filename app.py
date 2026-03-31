import gradio as gr
from utils.yolo_detector import PersonDetector

# Initialize detector
detector = PersonDetector("yolov8n.pt")

# Create Gradio interface with tabs
with gr.Blocks(title="YOLOv8 Person Detection", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# YOLOv8 Person Detection")
    gr.Markdown("Upload an image, video, or use your webcam to detect people in real-time")
    
    with gr.Tab("Image Upload"):
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="numpy", label="Upload Image")
                image_button = gr.Button("Detect People", variant="primary")
            with gr.Column():
                image_output = gr.Image(label="Detection Result")
                image_text = gr.Textbox(label="Detection Count", interactive=False)
        
        image_button.click(
            fn=detector.detect_image,
            inputs=image_input,
            outputs=[image_output, image_text]
        )
    
    with gr.Tab("Webcam (Real-time)"):
        gr.Markdown("Allow camera access when prompted")
        with gr.Row():
            with gr.Column():
                webcam_input = gr.Image(sources=["webcam"], streaming=True, label="Webcam Feed")
            with gr.Column():
                webcam_output = gr.Image(label="Detection Result")
                webcam_text = gr.Textbox(label="Detection Count", interactive=False)
        
        webcam_input.stream(
            fn=detector.detect_frame,
            inputs=webcam_input,
            outputs=[webcam_output, webcam_text]
        )
    
    with gr.Tab("Video Upload"):
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="Upload Video")
                video_button = gr.Button("Process Video", variant="primary")
            with gr.Column():
                video_output = gr.Image(label="First Frame Detection")
        
        video_button.click(
            fn=detector.detect_video,
            inputs=video_input,
            outputs=video_output
        )
    
    gr.Markdown("### About")
    gr.Markdown("""
    This demo uses the YOLOv8n (nano) model to detect people in images and videos.
    - **Model**: YOLOv8n from Ultralytics
    - **Detection**: Person class only (COCO dataset class 0)
    - **Output**: Bounding boxes with confidence scores
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)