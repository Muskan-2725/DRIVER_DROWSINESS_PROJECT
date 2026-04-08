import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torchvision.models import mobilenet_v2

model = mobilenet_v2(weights=None)

model.classifier = nn.Sequential(
    nn.Linear(1280, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(0.5),

    nn.Linear(256, 64),
    nn.BatchNorm1d(64),
    nn.ReLU(),
    nn.Dropout(0.3),

    nn.Linear(64, 1)
)
model.load_state_dict(torch.load("drowsiness_mobilenet.pth", map_location=device))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])


import cv2 as cv
import mediapipe as mp
import numpy as np
import threading
# import winsound
from face_direction import get_face_direction

from eye_detect import calculate_ear
from mouth_mar import calculate_mar

def crop_eye_from_landmarks(indices, landmarks, frame):
    h, w, _ = frame.shape
    pts = [(int(landmarks.landmark[i].x*w), int(landmarks.landmark[i].y*h)) for i in indices]

    x_vals = [p[0] for p in pts]
    y_vals = [p[1] for p in pts]

    x_min, x_max = max(min(x_vals)-5,0), min(max(x_vals)+5,w)
    y_min, y_max = max(min(y_vals)-5,0), min(max(y_vals)+5,h)

    return frame[y_min:y_max, x_min:x_max]

def predict_eye(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (224, 224))
    img = Image.fromarray(img)
    
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)              # ✅ FIRST compute output
        output = torch.sigmoid(output)     # ✅ THEN apply sigmoid
        pred = (output > 0.5).int()        # ✅ threshold

    return pred.item()
            

#  CONFIG 
FRAME_SKIP = 2
CALIB_FRAMES = 40
SLEEP_FRAMES = 6
YAWN_FRAMES = 5

#  INIT 
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Camera not opening")
    exit()

#  VARIABLES 
frame_id = 0
calibrated = False
frame_count = 0

calib_ear_values = []
calib_mar_values = []

baseline_ear = 0
baseline_mar = 0

sleep_counter = 0
yawn_counter = 0
alarm_on = False

EYE_CLOSED_FRAMES = 3
eye_closed_counter = 0

#  ALARM 
def play_alarm():
    if os.path.exists(ALARM_SOUND):
        os.system(f"afplay {ALARM_SOUND}")

#  LOOP 
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1

    if frame_id % FRAME_SKIP != 0:
        continue

    frame = cv.resize(frame, (640, 480))
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        direction = get_face_direction(face_landmarks, frame)

        ear, eye_pts = calculate_ear(face_landmarks, frame)
        LEFT_EYE_IDX = [33,133,160,159,158,144,153,154]
        RIGHT_EYE_IDX = [362,263,387,386,385,373,380,381]

        left_eye_img = crop_eye_from_landmarks(LEFT_EYE_IDX, face_landmarks, frame)
        right_eye_img = crop_eye_from_landmarks(RIGHT_EYE_IDX, face_landmarks, frame)
        if left_eye_img.size == 0 or right_eye_img.size == 0:
            continue

        pred_left = predict_eye(left_eye_img)
        pred_right = predict_eye(right_eye_img)
        if pred_left == 1 and pred_right == 1:
            eye_closed_counter += 1
        else:
            eye_closed_counter = 0
        print("Left:", pred_left, "Right:", pred_right)
        model_eye_closed = eye_closed_counter >= EYE_CLOSED_FRAMES
        mar, mouth_pts = calculate_mar(face_landmarks, frame)

        #  CALIBRATION 
        if not calibrated:
            calib_ear_values.append(ear)
            calib_mar_values.append(mar)
            frame_count += 1

            cv.putText(frame, f"Calibrating {frame_count}/{CALIB_FRAMES}",
                       (50,50), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

            if frame_count == CALIB_FRAMES:
                baseline_ear = max(calib_ear_values)
                baseline_mar = min(calib_mar_values)
                calibrated = True

                print("Baseline EAR:", baseline_ear)
                print("Baseline MAR:", baseline_mar)

            cv.imshow("Drowsiness Detection", frame)
            if cv.waitKey(1) == 27:
                break
            continue

        #  THRESHOLDS 
        EAR_THRESHOLD = baseline_ear * 0.8
        MAR_THRESHOLD = baseline_mar + 0.25

        # DETECTION 
        if direction == "FORWARD":

            if ear < EAR_THRESHOLD and model_eye_closed:
                sleep_counter += 1
            else:
                sleep_counter = 0

            if mar > MAR_THRESHOLD:
                yawn_counter += 1
            else:
                yawn_counter = 0

        else:
            sleep_counter = 0
            yawn_counter = 0

            cv.putText(frame, "LOOK STRAIGHT!", (50,100),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        #  ALERT 
        if sleep_counter > SLEEP_FRAMES:
            cv.putText(frame, "DROWSY!", (50,100),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

            if not alarm_on:
                alarm_on = True
                threading.Thread(target=play_alarm, daemon=True).start()

        elif yawn_counter > YAWN_FRAMES:
            cv.putText(frame, "YAWNING!", (50,100),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

            if not alarm_on:
                alarm_on = True
                threading.Thread(target=play_alarm, daemon=True).start()
        else:
            alarm_on = False

        #  DRAW 
        for p in eye_pts:
            cv.circle(frame, p, 2, (0,255,0), -1)

        for p in mouth_pts:
            cv.circle(frame, p, 2, (255,0,0), -1)

        #  DEBUG
        cv.putText(frame, f"EAR: {ear:.2f} TH:{EAR_THRESHOLD:.2f}", (50,150),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv.putText(frame, f"MAR: {mar:.2f} TH:{MAR_THRESHOLD:.2f}", (50,180),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
        cv.putText(frame, f"Face: {direction}", (50, 220),
           cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

    else:
        cv.putText(frame, "No face detected", (50,50),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    cv.imshow("Drowsiness Detection", frame)

    if cv.waitKey(1) == 27:
        break

cap.release()
cv.destroyAllWindows()