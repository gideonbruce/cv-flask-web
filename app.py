from flask import Flask, request, render_template, jsonify, Response
import cv2
import uuid
import mysql.connector
from ultralytics import YOLO
import threading
import time
import matplotlib.pyplot as plt
import base64
from io import BytesIO 
import os
from PIL import Image
from flask_bcrypt import Bcrypt
from flask_wtf.csrf import CSRFProtect
from forms import SignupForm, LoginForm
from detection_utils import get_most_frequent_bboxes, generate_bboxes_plot

app = Flask(__name__)

app.secret_key = 'ifhutehrfjsxmfr'
bcrypt = Bcrypt(app)
csrf = CSRFProtect

# Connecting to MySQL
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="@Gideon",
    database="weed_detections"
)
cursor = db.cursor()

#signup route

#route to display most frequent bounding boxes
@app.route('/most_frequent_bboxes')
def most_frequent_bboxes():
    most_common_boxes = get_most_frequent_bboxes()

    img_base64 = generate_bboxes_plot(most_common_boxes)

    return render_template('most_frequent_bboxes.html',plot_data=img_base64)

#route to display plotting
@app.route('/detections_over_time')
def detections_over_time():
    img_base64 = generate_detection_plot()

    return render_template('detections_over_time.html', plot_data=img_base64)

def generate_detection_plot():
    cursor.execute("SELECT DATE(detected_on) AS date, COUNT(*) AS daily_detections FROM detections GROUP BY DATE(detected_on) ORDER BY date")
    temporal_data = cursor.fetchall()

    #preparing data for plotting
    dates = [row[0] for row in temporal_data]
    detections = [row[1] for row in temporal_data]

    #creating plot using matplotlib
    plt.figure(figsize=(10, 6))
    plt.plot(dates, detections, marker='o', color='b', label='Daily detections')
    plt.title("Daily detections overtime")
    plt.xlabel("Date")
    plt.ylabel("Number of detections")
    plt.xticks(rotation=45)
    plt.tight_layout()

    # saving the plot as an image in base64 format
    img= BytesIO()
    plt.savefig(img, format='png')
    img.seek(0) 
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    return img_base64

    

# Load YOLOv8 model
model = YOLO("best.pt")

# Set up OpenCV video capture (webcam)
cap = cv2.VideoCapture(0)

# Function to run live detection
def detect_live():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Running YOLO detection on the frame
        results = model(frame)
        detections = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls = result.names[int(box.cls[0])]
                detections.append({"class": cls, "bbox": [x1, y1, x2, y2]})

                # Drawing bounding boxes on the frame
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, cls, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Save detection data to MySQL
                cursor.execute("INSERT INTO detections (class, x1, y1, x2, y2) VALUES (%s, %s, %s, %s, %s)",
                               (cls, x1, y1, x2, y2))
                db.commit()

        # Encode the frame to send to the browser
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        time.sleep(3)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Start live detection route
@app.route('/start_live_detection')
def start_live_detection():
    return Response(detect_live(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
