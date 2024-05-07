# FINAL GRADIO MODEL CODE

import PIL.Image as Image
import gradio as gr
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("version4c.pt")

# Function to predict image
def predict_image(img, conf_threshold, iou_threshold):
    # Perform object detection
    results = model.predict(
        source=img,
        conf=conf_threshold,
        iou=iou_threshold,
        show_labels=True,
        show_conf=True,
        imgsz=640,
    )

    # Plot the result
    for r in results:
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])

    return im

# Interface configuration
iface = gr.Interface(
    fn=predict_image,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence threshold"),
        gr.Slider(minimum=0, maximum=1, value=0.45, label="IoU threshold")
    ],
    outputs=gr.Image(type="pil", label="Result"),
    title="Ultralytics Gradio",
    description="Upload images for inference. The Ultralytics YOLOv8n model is used by default."
)

# Launch the interface without specifying host and port
if __name__ == '__main__':
    iface.launch()

