import cv2
import os
import face_recognition
from datetime import datetime, timedelta
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import numpy as np

# Output folders
output_folder = "detected_faces_and_persons"
log_file_path = "reappearance_log1.txt"
time_log_path = "in_out_log1.txt"

# Cleanup and setup
if os.path.exists(output_folder):
    for file in os.listdir(output_folder):
        os.remove(os.path.join(output_folder, file))
    os.rmdir(output_folder)

os.makedirs(output_folder, exist_ok=True)

# Load YOLOv8 model
model = YOLO("yolov8n.pt")
model.classes = [0]  # Detecting persons

# Load video
video_path = "f1.mp4"  # Replace with your video path
cap = cv2.VideoCapture(video_path)

# Video properties
frame_rate = cap.get(cv2.CAP_PROP_FPS) or 30
frame_duration = 1 / frame_rate
base_time = datetime.strptime("10:00:00.000", "%H:%M:%S.%f")

# Initialization
frame_count = 0
face_count = 0
known_face_encodings = []
reappearance_data = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    current_time = frame_count * frame_duration
    video_time = base_time + timedelta(seconds=current_time)
    video_time_str = video_time.strftime("%H:%M:%S.%f")[:-3]

    # YOLO person detection
    results = model(frame)
    detections = results[0].boxes

    annotator = Annotator(frame)

    for box in detections:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        person_image = frame[y1:y2, x1:x2]

        # Face recognition within detected person
        rgb_person = cv2.cvtColor(person_image, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_person,model='cnn')
        face_encodings = face_recognition.face_encodings(rgb_person, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)

            if not any(matches):
                known_face_encodings.append(face_encoding)
                face_id = face_count
                face_count += 1
                face_path = os.path.join(output_folder, f"person_{face_count}.jpg")
                cv2.imwrite(face_path, person_image)
            else:
                face_id = matches.index(True)

            # Logging
            reappearance_data.setdefault(face_id, {"frames": [], "times": []})
            reappearance_data[face_id]["frames"].append(frame_count)
            reappearance_data[face_id]["times"].append(video_time_str)

            # Draw bounding boxes
            top, right, bottom, left = face_location
            cv2.rectangle(person_image, (left, top), (right, bottom), (0, 255, 0), 2)
            annotator.box_label([x1, y1, x2, y2], f"Person {face_id}", color=(0, 255, 0))

    # Display
    cv2.imshow("Detection & Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# Save logs
with open(log_file_path, "w") as log_file:
    for face_id, data in reappearance_data.items():
        log_file.write(f"ID: {face_id}\nFrames: {data['frames']}\nTimes: {data['times']}\n{'-'*50}\n")

with open(time_log_path, "w") as time_log:
    for face_id, data in reappearance_data.items():
        time_log.write(f"ID: {face_id}\nIn Time: {len(data['frames'])}\nOut Time: {frame_count - len(data['frames'])}\n{'-'*50}\n")

