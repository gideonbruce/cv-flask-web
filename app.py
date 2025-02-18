from flask import Flask, request, render_template, jsonify
import os
import cv2
import uuid
import mysql.connector
from ultralytics import YOLO
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Connecting to MySQL
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="@Gideon",
    database="weed_detection_db"
)
cursor = db.cursor()

# Load YOLOv8 model
model = YOLO("best.pt") 

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Upload and detect route
@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"})
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    # Save image
    filename = str(uuid.uuid4()) + '.jpg'
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Run YOLO detection
    img = Image.open(filepath)
    results = model(img)
    
    weed_count = 0  # Initialize weed count

    # Iterate over detections and count weeds
    detections = []
    for result in results:
        for box in result.boxes:
            cls = result.names[int(box.cls[0])]
            if cls == 'weed':  # Assuming 'weed' is one of the classes detected by YOLO
                weed_count += 1
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({"class": cls, "bbox": [x1, y1, x2, y2]})

    # Save the weed count and image path to MySQL
    cursor.execute("INSERT INTO images (file_path, weed_count) VALUES (%s, %s)", (filename, weed_count))
    db.commit()
    
    return jsonify({"detections": detections, "weed_count": weed_count, "image": filename})

if __name__ == '__main__':
    app.run(debug=True)
