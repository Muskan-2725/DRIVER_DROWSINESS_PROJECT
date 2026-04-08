import streamlit as st
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
from torchvision import transforms
import cv2 as cv
import numpy as np
from PIL import Image
import mediapipe as mp

from eye_detect import calculate_ear
from mouth_mar import calculate_mar
from face_direction import get_face_direction

#  PAGE CONFIG 
st.set_page_config(page_title="Drowsiness Detection", layout="wide")
st.title("🚗 Driver Drowsiness Detection System")

#  LOAD MODEL 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

#  TRANSFORM 
transform = transforms.Compose([
    transforms.Resize((224,224)),  # MATCH TRAINING
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

#  MEDIAPIPE 
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)

LEFT_EYE_IDX = [33,133,160,159,158,144,153,154]
RIGHT_EYE_IDX = [362,263,387,386,385,373,380,381]

def crop_eye(indices, landmarks, frame):
    h, w, _ = frame.shape
    pts = [(int(landmarks.landmark[i].x*w), int(landmarks.landmark[i].y*h)) for i in indices]

    x_vals = [p[0] for p in pts]
    y_vals = [p[1] for p in pts]

    # 🔥 increase margin (CRUCIAL)
    margin = 15

    x_min = max(min(x_vals) - margin, 0)
    x_max = min(max(x_vals) + margin, w)
    y_min = max(min(y_vals) - margin, 0)
    y_max = min(max(y_vals) + margin, h)

    return frame[y_min:y_max, x_min:x_max]

def predict_eye(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        prob = torch.sigmoid(output).item()

    return prob

#  SIDEBAR 
mode = st.sidebar.radio("Select Mode", ["Webcam", "Upload Image"])
# threshold = st.sidebar.slider("Drowsy Threshold", 0.3, 0.9, 0.5)

# WEBCAM

if mode == "Webcam":

    run = st.checkbox("Start Webcam")

    frame_window = st.image([])
    status_text = st.empty()

    if run:
        cap = cv.VideoCapture(0)

        # VARIABLES
        frame_id = 0
        calibrated = False
        frame_count = 0

        calib_ear_values = []
        calib_mar_values = []

        baseline_ear = 0
        baseline_mar = 0

        sleep_counter = 0
        yawn_counter = 0

        EYE_CLOSED_FRAMES = 3
        eye_closed_counter = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv.resize(frame, (640, 480))
            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]

                # ===== FACE DIRECTION =====
                direction = get_face_direction(face_landmarks, frame)

                # ===== EAR + MAR =====
                ear, eye_pts = calculate_ear(face_landmarks, frame)
                mar, mouth_pts = calculate_mar(face_landmarks, frame)

                # ===== EYE CROPPING =====
                left_eye = crop_eye(LEFT_EYE_IDX, face_landmarks, frame)
                right_eye = crop_eye(RIGHT_EYE_IDX, face_landmarks, frame)

                if left_eye.size != 0 and right_eye.size != 0:
                    pred_left = predict_eye(left_eye)
                    pred_right = predict_eye(right_eye)

                    if pred_left > 0.5 and pred_right > 0.5:
                        eye_closed_counter += 1
                    else:
                        eye_closed_counter = 0

                # ===== CALIBRATION =====
                if not calibrated:
                    calib_ear_values.append(ear)
                    calib_mar_values.append(mar)
                    frame_count += 1

                    status_text.warning(f"Calibrating {frame_count}/40")

                    if frame_count == 40:
                        baseline_ear = max(calib_ear_values)
                        baseline_mar = min(calib_mar_values)
                        calibrated = True

                    frame_window.image(rgb)
                    continue

                # ===== THRESHOLDS (FIXED) =====
                EAR_THRESHOLD = baseline_ear * 0.8
                MAR_THRESHOLD = max(0.5, baseline_mar + 0.15)  # 🔥 FIXED

                # ===== DETECTION =====
                if direction == "FORWARD":

                    # DROWSINESS
                    if ear < EAR_THRESHOLD and eye_closed_counter >= EYE_CLOSED_FRAMES:
                        sleep_counter += 1
                    else:
                        sleep_counter = 0

                    # YAWNING (IMPROVED)
                    if mar > MAR_THRESHOLD:
                        yawn_counter += 1
                    else:
                        yawn_counter = max(0, yawn_counter - 1)

                else:
                    sleep_counter = 0
                    yawn_counter = 0

                    cv.putText(frame, "LOOK STRAIGHT!", (50,100),
                               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

                # ===== STATUS =====
                if sleep_counter > 6:
                    status_text.error("😴 DROWSY")
                    cv.putText(frame, "DROWSY!", (50,100),
                               cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

                elif yawn_counter > 3:
                    status_text.warning("😮 YAWNING")
                    cv.putText(frame, "YAWNING!", (50,100),
                               cv.FONT_HERSHEY_SIMPLEX, 1, (0,165,255), 3)

                else:
                    status_text.success("👀 AWAKE")

                # ===== DRAW LANDMARKS =====
                for p in eye_pts:
                    cv.circle(frame, p, 2, (0,255,0), -1)

                for p in mouth_pts:
                    cv.circle(frame, p, 2, (255,0,0), -1)

                # ===== DEBUG INFO =====
                cv.putText(frame, f"EAR: {ear:.2f} TH:{EAR_THRESHOLD:.2f}", (50,150),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                cv.putText(frame, f"MAR: {mar:.2f} TH:{MAR_THRESHOLD:.2f}", (50,180),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

                cv.putText(frame, f"Face: {direction}", (50,220),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

            else:
                status_text.warning("No face detected")

            frame_window.image(cv.cvtColor(frame, cv.COLOR_BGR2RGB))

        cap.release()

## upload image mode
def predict_eye_upload(img):
    # Convert BGR → GRAYSCALE (MATCH TRAINING)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Resize EXACTLY as training
    img = cv.resize(img, (224, 224))

    # Convert back to 3-channel (fake RGB)
    img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)

    img = Image.fromarray(img)
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        prob = torch.sigmoid(output).item()

    return prob

if mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload Face Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        frame = np.array(image)

        st.image(frame, caption="Uploaded Image", use_column_width=True)

        rgb = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        results = face_mesh.process(cv.cvtColor(frame, cv.COLOR_RGB2BGR))

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]

            left_eye = crop_eye(LEFT_EYE_IDX, landmarks, frame)
            right_eye = crop_eye(RIGHT_EYE_IDX, landmarks, frame)

            if left_eye.size != 0 and right_eye.size != 0:

                prob_left = predict_eye_upload(left_eye)
                prob_right = predict_eye_upload(right_eye)

                avg_prob = (prob_left + prob_right) / 2

                st.write(f"Left Prob: {prob_left:.2f}, Right Prob: {prob_right:.2f}")

                if avg_prob > 0.7:
                    st.error(f"😴 DROWSY | Confidence: {avg_prob:.2f}")
                else:
                    st.success(f"👀 AWAKE | Confidence: {avg_prob:.2f}")

                # Optional debug
                # st.write(f"Left Eye: {pred_left}, Right Eye: {pred_right}")
                st.image(left_eye, caption="Left Eye", width=150)
                st.image(right_eye, caption="Right Eye", width=150)

            else:
                st.warning("Eyes not detected properly")

        else:
            st.error("No face detected")