import argparse
import os
import datetime
import base64
import cv2
import numpy as np
from flask import Flask, render_template, request, Response
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load the YOLOv9 model
model = YOLO('version4.pt')  # Replace with the actual path to your model

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            file_extension = file.filename.rsplit('.', 1)[1].lower()

            if file_extension in ['jpg', 'jpeg', 'png', 'bmp']:
                img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
                detections = model(img)
                image_path, encoded_image = process_image(file.filename, detections)
                return render_template('index.html', image_path=image_path, encoded_image=encoded_image)

            elif file_extension == 'mp4':
                video_path = save_file(file)
                return render_template('index.html', video_path=video_path)

    return render_template('index.html')

def process_image(filename, detections):
    directory = 'runs/detect'
    if not os.path.exists(directory):
        os.makedirs(directory)

    subfolder_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    subfolder_path = os.path.join(directory, subfolder_name)
    os.makedirs(subfolder_path)

    image_path = os.path.join(subfolder_path, f"detected_{filename}")
    detections[0].save(image_path)

    with open(image_path, 'rb') as f:
        image_data = f.read()
        encoded_image = base64.b64encode(image_data).decode('utf-8')

    return image_path, encoded_image

def save_file(file):
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    return file_path

def gen_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Change the codec if needed
    out = cv2.VideoWriter('output.avi', fourcc, 30.0, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            results = model(frame)
            res_plotted = results[0].plot()
            out.write(res_plotted)
            ret, buffer = cv2.imencode('.jpg', res_plotted)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.1)
    finally:
        cap.release()
        out.release()

@app.route('/video_feed')
def video_feed():
    video_path = request.args.get('video_path')
    if video_path:
        return Response(gen_frames(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "No video path provided"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing YOLOv9 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=args.port, debug=True)