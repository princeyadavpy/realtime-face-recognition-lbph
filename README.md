# Real-Time Face Recognition using LBPH

Python script for real-time face detection and recognition using OpenCV (LBPH).  
Supports capturing face samples, training, and verifying a person via webcam.

## Features
- Detect faces using Haar Cascade
- Capture face samples and save to `faces_dataset/`
- Train LBPH face recognizer
- Real-time recognition with name + confidence
- Verify a specific person (`v` key)

## Requirements
pip install opencv-contrib-python numpy

Run the script:


python face_recognition.py
Controls:

c → Capture new face samples

v → Verify a person

q → Quit

Notes
Faces are stored in faces_dataset/<person_name>/.

Haar cascade is loaded from OpenCV's default path.

Author
Prince Yadav
