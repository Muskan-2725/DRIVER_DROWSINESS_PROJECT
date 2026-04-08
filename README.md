# DRIVER DROWSINESS DETECTION SYSTEM
## Overview

The Driver Drowsiness Detection System is an AI-based application designed to detect signs of driver fatigue in real-time and prevent potential accidents. The system uses computer vision and deep learning techniques to monitor eye movements and alert the driver when drowsiness is detected.

## Objective

To enhance road safety by identifying early signs of drowsiness and providing timely alerts to the driver.

## Key Features
- Real-time eye detection using webcam
- Deep learning model for drowsiness prediction
- Alert system when drowsiness is detected
- Image upload support for testing
- Lightweight and easy-to-use interface

## Tech Stack
- Programming Language: Python
- Libraries & Frameworks: OpenCV, NumPy, Matplotlib
- Model Type: Convolutional Neural Network (CNN)
- Tools: VS Code, Git, GitHub

## How It Works
1. Captures live video stream using webcam
2. Detects face and extracts eye region
3. Preprocesses the eye image (grayscale, resize, RGB conversion)
4. Passes the image to the trained CNN model
5. Predicts whether eyes are open or closed
6. Triggers alert if drowsiness is detected

## Installation & Setup
1. Clone the repository
    - git clone https://github.com/Muskan-2725/DRIVER_DROWSINESS_PROJECT
    - cd DRIVER_DROWSINESS
2. Install dependencies
    - pip install -r requirements.txt
3. Run the application
    - python app.py

## Results
- Achieved high accuracy in detecting drowsiness using CNN
- Works effectively in real-time webcam scenarios
- Successfully identifies closed-eye patterns

## GitHub Repository
LINK : https://github.com/Muskan-2725/DRIVER_DROWSINESS_PROJECT
