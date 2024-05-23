from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import os
import torch
import random


# cap = cv2.VideoCapture(1)  # For Webcam
# cap.set(3, 1280)
# cap.set(4, 720)

cap = cv2.VideoCapture("../Object_Detection_Project/Videos/potholes_vid.mp4")  # For Video
model = YOLO("../Yolo-Weights/potholes_model.pt")
classNames = ["pothole"]
imageBackground = cv2.imread("images/image_background_pothole.png")
prev_frame_time = 0
new_frame_time = 0


# Create Directory
os.makedirs("pothole_images", exist_ok=True)
os.makedirs("pothole_coordiationes", exist_ok=True)
pothole_count = 0

# Define the scale percentage for resizing
scale_percent = 55  # You can adjust this value as needed
# Example coordinates and size
x, y, w, h = 100, 100, 524, 278  # replace with your actual values
def resize_frame(image, width, height):
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

def get_dummy_gps_coors():
    # Generate Random long and lat coordinates
    lat = random.uniform(-90, 90)
    lon = random.uniform(-180, 180)
    return lat, lon

while True:
    new_frame_time = time.time()
    success, img = cap.read()
    if not success:
        break

        # Resize the frame from the video to fit the designated area
    resized_img = resize_frame(img, w, h)

    # Overlay the resized frame onto the background image
    imageBackground[y:y + h, x:x + w] = resized_img
    # Resize the frame
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])

            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

            # Check if the confidence above of below the threshold
    if conf > 0.7:
        # Cope the Pothole image and save it
        pothole_image = img[y1:y2, x1:x2]
        cv2.imwrite(f"pothole_images/pothole_{pothole_count}.jpg", pothole_image)

        # Get the dummy gps coordinates of the pothole
        lat, lon = get_dummy_gps_coors()
        with open(f"pothole_coordiationes/pothole_{pothole_count}.txt", 'w') as f:
            f.write(f"Latitude: {lat}, Longitude: {lon}")

        pothole_count += 1
    cv2.imshow("Video", img)
    cv2.imshow("Potholes Detection",imageBackground)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# from ultralytics import YOLO
# import cv2
#
# import cvzone
# import math
# import time
# import os
#
#
# # Load the YOLO model
# model = YOLO("../Yolo-Weights/potholes_model.pt")
# classNames = ["pothole"]
#
# # Create Directory
# os.makedirs("pothole_images", exist_ok=True)
# os.makedirs("pothole_coordinationes", exist_ok=True)
#
# # Open the camera
# cap = cv2.VideoCapture(0)  # 0 for the default camera
#
# # Define the scale percentage for resizing
# scale_percent = 50  # You can adjust this value as needed
#
# while True:
#     # Read a frame from the camera
#     ret, frame = cap.read()
#
#     # Resize the frame
#     width = int(frame.shape[1] * scale_percent / 100)
#     height = int(frame.shape[0] * scale_percent / 100)
#     dim = (width, height)
#     resized_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
#
#     # Perform object detection
#     results = model(resized_frame, stream=True)
#
#     for r in results:
#         boxes = r.boxes
#         for box in boxes:
#             # Bounding Box
#             x1, y1, x2, y2 = box.xyxy[0].int().tolist()
#             w, h = x2 - x1, y2 - y1
#             cvzone.cornerRect(frame, (x1, y1, w, h))
#
#             # Confidence
#             conf = math.ceil((box.conf[0] * 100)) / 100
#             # Class Name
#             cls = int(box.cls[0])
#
#             cvzone.putTextRect(frame, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
#
#             # Check if the confidence above or below the threshold
#             if conf > 0.5:
#                 # Copy the pothole image and save it
#                 pothole_image = frame[y1:y2, x1:x2]
#                 cv2.imwrite(f"pothole_images/pothole_{time.time()}.jpg", pothole_image)
#
#     # Display the frame
#     cv2.imshow("Pothole Detection", frame)
#
#     # Check for key press to exit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # Release the camera and close all windows
# cap.release()
# cv2.destroyAllWindows()
