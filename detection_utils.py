# detection_utils.py

import mysql.connector
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO
import base64

# Database connection
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="@Gideon",
        database="weed_detection_db"
    )

# Function to fetch the most frequent bounding boxes from the database
def get_most_frequent_bboxes():
    db = get_db_connection()
    cursor = db.cursor()

    # Query to fetch bounding box coordinates from the database
    cursor.execute("SELECT x1, y1, x2, y2 FROM detections")

    # Fetch all bounding box coordinates
    bounding_boxes = cursor.fetchall()

    # Use Counter to find the most frequent bounding boxes
    counter = Counter(bounding_boxes)

    # Get the most common bounding boxes (for example, top 5)
    most_common_boxes = counter.most_common(5)

    return most_common_boxes

# Function to generate a plot of the most frequent bounding boxes
def generate_bboxes_plot(most_common_boxes):
    # Create a plot to visualize the bounding boxes
    fig, ax = plt.subplots()

    # Example image: Replace with the actual image you're using to draw boxes on
    img = plt.imread('sample_image.jpg')  # You can replace it with any image or background
    ax.imshow(img)

    # Draw the most frequent bounding boxes
    for bbox, count in most_common_boxes:
        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.title("Most Frequent Bounding Boxes")

    # Convert plot to base64 to send to the HTML page
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    return img_base64
