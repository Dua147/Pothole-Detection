
from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import os
import random
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import webbrowser

# Set up video capture
cap = cv2.VideoCapture("../Object_Detection_Project/Videos/Potholes_Vid_Trim.mp4")  # For Video
model = YOLO("../Yolo-Weights/potholes_model.pt")
classNames = ["pothole"]
prev_frame_time = 0
new_frame_time = 0

# Create directories
os.makedirs("pothole_images", exist_ok=True)
os.makedirs("pothole_coordinationes", exist_ok=True)
pothole_count = 0

# Define the scale percentage for resizing
scale_percent = 55  # You can adjust this value as needed

# Parameters for tracking
iou_threshold = 0.3
trackers = []

def resize_frame(image, width, height):
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

def get_dummy_gps_coors():
    # Generate random long and lat coordinates
    lat = random.uniform(-90, 90)
    lon = random.uniform(-180, 180)
    return lat, lon

def open_pothole_coordinates():
    path = os.path.abspath("pothole_coordinationes")
    webbrowser.open(f'file://{path}')

# Non-maximum suppression to filter overlapping boxes
def non_max_suppression(boxes, threshold=0.3):
    if len(boxes) == 0:
        return boxes

    # Sort boxes by confidence score in descending order
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    picked_boxes = []

    while boxes:
        chosen_box = boxes.pop(0)
        picked_boxes.append(chosen_box)
        boxes = [box for box in boxes if iou(chosen_box, box) < threshold]

    return picked_boxes

# Intersection over Union (IoU) calculation
def iou(box1, box2):
    x1, y1, x2, y2 = box1[:4]
    x1_, y1_, x2_, y2_ = box2[:4]

    xi1 = max(x1, x1_)
    yi1 = max(y1, y1_)
    xi2 = min(x2, x2_)
    yi2 = min(y2, y2_)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_ - x1_) * (y2_ - y1_)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area

def is_new_pothole(new_box, trackers, iou_threshold):
    for tracker in trackers:
        if iou(new_box, tracker) > iou_threshold:
            return False
    return True

# Set up Tkinter
root = tk.Tk()
root.title("Pothole Detection")

# Create a frame for the video and controls
frame = ttk.Frame(root)
frame.pack(fill=tk.BOTH, expand=True)

# Create a label to display the video frames
label = tk.Label(frame)
label.pack(side=tk.RIGHT)

# Create a frame for the controls (quit button, pothole count, and link)
control_frame = ttk.Frame(frame)
control_frame.pack(side=tk.LEFT, padx=10, pady=10)

# Create a button to quit the application
quit_button = tk.Button(control_frame, text="Quit", command=root.quit, bg="#0E46A3", fg="white", font=("Arial", 16, "bold"))
quit_button.pack(side=tk.TOP, pady=5)

# Create a label to display the pothole count inside a light blue placeholder
pothole_placeholder = tk.Frame(control_frame, bg="lightblue", padx=5, pady=5)
pothole_placeholder.pack(side=tk.TOP, pady=5)

pothole_label = tk.Label(pothole_placeholder, text="Potholes detected: 0", bg="lightblue", font=("Arial", 12))
pothole_label.pack()

# Create an underlined label to check pothole locations
location_label = tk.Label(root, text="Check Potholes Location", fg="blue", cursor="hand2", font=("Arial", 10, "underline"))
location_label.pack(side=tk.BOTTOM, pady=10)
location_label.bind("<Button-1>", lambda e: open_pothole_coordinates())

def update_frame():
    global pothole_count
    global trackers

    new_frame_time = time.time()
    success, img = cap.read()
    if not success:
        cap.release()
        root.quit()
        return

    # Resize the frame
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    results = model(img, stream=True)
    boxes = []

    for r in results:
        for box in r.boxes:
            # Bounding Box and Confidence
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])

            if conf > 0.5:  # Adjusted confidence threshold
                boxes.append((x1, y1, x2, y2, conf, cls))

    # Apply non-maximum suppression
    filtered_boxes = non_max_suppression(boxes)

    for (x1, y1, x2, y2, conf, cls) in filtered_boxes:
        # drawing the bounding boxes
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h))
        cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

        if is_new_pothole((x1, y1, x2, y2, conf, cls), trackers, iou_threshold):
            trackers.append((x1, y1, x2, y2, conf, cls))

            # Save the detected pothole image and coordinates
            pothole_image = img[y1:y2, x1:x2]
            cv2.imwrite(f"pothole_images/pothole_{pothole_count}.jpg", pothole_image)
            lat, lon = get_dummy_gps_coors()
            with open(f"pothole_coordinationes/pothole_{pothole_count}.txt", 'w') as f:
                f.write(f"Latitude: {lat}, Longitude: {lon}")

            pothole_count += 1
            pothole_label.config(text=f"Potholes detected: {pothole_count}")

    # Converting the image to RGB (Tkinter requires images in RGB format)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(image=img_pil)

    # Update the label with the new frame
    label.img_tk = img_tk
    label.config(image=img_tk)

    # Schedule the next frame update
    root.after(10, update_frame)

# Start updating frames
update_frame()

# Run the Tkinter event loop
root.mainloop()

