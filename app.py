from flask import Flask, request, render_template, jsonify, Response
import cv2
import uuid
import mysql.connector
from ultralytics import YOLO
import threading
import time

app = Flask(__name__)

# Connecting to MySQL
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="@Gideon",
    database="weed_detections"
)
cursor = db.cursor()

#route to display plotting
@app.route('/detections_over_time')
def detections_over_time():
    img_base64 = generate_detection_plot()

    return render_template('detections_over_time.html', plot_data=img_base64)

def generate_detection_plot():
    cursor.execute("SELECT DATE(detected_on) AS date, COUNT(*) AS daily_detections FROM detections GROUP BY DATE(detected_on) ORDER BY date")

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
