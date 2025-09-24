from flask import Flask, render_template, Response
import cv2
import numpy as np
import pygame
import os
import dlib
from scipy.spatial import distance
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__)

# Initialize pygame for alarm sound
pygame.mixer.init()

# Path to alarm sound file
alarm_sound_path = "static/alarm.mp3"
if os.path.exists(alarm_sound_path):
    pygame.mixer.music.load(alarm_sound_path)
else:
    print(f"Warning: Alarm sound file '{alarm_sound_path}' not found!")

# Path to model file
model_path = "drowsiness_cnn_model.h5"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Error: Model file '{model_path}' not found! Please check the path.")

# Load the model
model = load_model(model_path)

# Initialize dlib's face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Eye landmark positions
(lStart, lEnd) = (42, 48)
(rStart, rEnd) = (36, 42)

# Drowsiness detection parameters
eye_threshold = 0.25
eye_frame_limit = 20  # Number of frames to trigger alarm
frame_counter = 0

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Function to generate video frames
def generate_frames():
    global frame_counter
    cap = cv2.VideoCapture(0)  # Open the webcam

    if not cap.isOpened():
        print("Error: Unable to access the camera!")
        return

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            for face in faces:
                shape = predictor(gray, face)
                shape = np.array([[p.x, p.y] for p in shape.parts()])

                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]

                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0

                if ear < eye_threshold:
                    frame_counter += 1
                    if frame_counter >= eye_frame_limit:
                        pygame.mixer.music.play()
                        cv2.putText(frame, "DROWSINESS ALERT!", (50, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                else:
                    frame_counter = 0
                    pygame.mixer.music.stop()

                # Draw landmarks on the eyes
                for (x, y) in np.concatenate((leftEye, rightEye), axis=0):
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    finally:
        cap.release()
        pygame.mixer.quit()  # Properly quit pygame

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
