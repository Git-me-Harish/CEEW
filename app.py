import argparse
import io
from PIL import Image
import datetime
import cv2
import numpy as np
from re import DEBUG, sub
from flask import Flask, render_template, request, redirect, send_file, url_for, Response
from werkzeug.utils import secure_filename
import os
import base64

from ultralytics import YOLO

app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route("/", methods=["GET", "POST"])
def predict_img():
    if request.method == "POST":
        if 'file' in request.files:
            f = request.files['file']
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(basepath, 'uploads', f.filename)
            print(f"upload folder is {filepath}")
            f.save(filepath)
            global imgpath
            predict_img.imgpath = f.filename
            print(f"printing predict_img: {predict_img}")

            file_extension = f.filename.rsplit('.', 1)[1].lower()

            if file_extension == 'jpg':
                img = cv2.imread(filepath)

                # Perform the detection
                model = YOLO('version4.pt')
                detections = model(img)
                return display(f.filename, detections)

            elif file_extension == 'mp4':
                video_path = filepath  # replace with your video path
                cap = cv2.VideoCapture(video_path)

                # get video dimensions
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                # Define the codec and create VideoWriter object
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (frame_width, frame_height))

                # initialize the YOLOv8 model here
                model = YOLO('version4.pt')

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # do YOLOv9 detection on the frame here
                    results = model(frame)
                    print(results)
                    cv2.waitKey(1)

                    res_plotted = results[0].plot()
                    cv2.imshow("result", res_plotted)

                    # write the frame to the output video
                    out.write(res_plotted)

                    if cv2.waitKey(1) == ord('q'):
                        break

                return video_feed()

    return render_template('index.html', image_path=None)

# The display function is used to serve the image or video from the folder_path directory.
@app.route('/<path:filename>', methods=['GET'])
def display(filename, detections=None):
    print(f"printing filename: {filename}")

    file_extension = filename.rsplit('.', 1)[1].lower()

    if file_extension in ['jpg', 'jpeg', 'png', 'bmp']:
        if detections is not None:
            image_data = detections[0].plot()
            _, encoded_image = cv2.imencode('.jpg', image_data)
            encoded_image = base64.b64encode(encoded_image).decode('utf-8')
            return render_template('index.html', encoded_image=encoded_image)
        else:
            return "No detections found"
    else:
        return "Invalid file format"

def get_frame():
    folder_path = os.getcwd()
    mp4_files = 'output.mp4'
    video = cv2.VideoCapture(mp4_files)  # detected video path
    while True:
        success, image = video.read()
        if not success:
            break
        ret, jpeg = cv2.imencode('.jpg', image)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        time.sleep(0.1)  # control the frame rate to display one frame every 100 milliseconds

# function to display the detected objects video on html page
@app.route("/video_feed")
def video_feed():
    print("function called")
    return Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov9 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    model = YOLO('version4.pt')
    app.run(host="0.0.0.0", port=args.port, debug=True)