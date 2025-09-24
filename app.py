from flask import Flask, render_template, Response
import cv2
import numpy as np
import pygame
import threading
import time
import mediapipe as mp
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app)

# Initialize pygame for alarm sound
pygame.mixer.init()
pygame.mixer.music.load("static/alarm.mp3")

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Constants
EYE_AR_THRESH = 0.2  # Aspect ratio threshold for closed eyes
EYE_FRAME_LIMIT = 20  # Consecutive frames to trigger alarm
frame_counter = 0
alarm_active = False
lock = threading.Lock()

def eye_aspect_ratio(landmarks, eye_points):
    # Calculate Euclidean distances between the two sets of vertical eye landmarks
    A = np.linalg.norm(np.array([landmarks[eye_points[1]]]) - np.array([landmarks[eye_points[5]]]))
    B = np.linalg.norm(np.array([landmarks[eye_points[2]]]) - np.array([landmarks[eye_points[4]]]))
    C = np.linalg.norm(np.array([landmarks[eye_points[0]]]) - np.array([landmarks[eye_points[3]]]))
    ear = (A + B) / (2.0 * C)
    return ear

def generate_frames():
    global frame_counter, alarm_active
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Extract eye landmarks
                landmarks = {i: (int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) 
                             for i, lm in enumerate(face_landmarks.landmark)}
                
                left_eye = [33, 160, 158, 133, 153, 144]
                right_eye = [362, 385, 387, 263, 373, 380]
                
                left_EAR = eye_aspect_ratio(landmarks, left_eye)
                right_EAR = eye_aspect_ratio(landmarks, right_eye)
                avg_EAR = (left_EAR + right_EAR) / 2.0
                
                if avg_EAR < EYE_AR_THRESH:
                    frame_counter += 1
                    if frame_counter >= EYE_FRAME_LIMIT and not alarm_active:
                        with lock:
                            alarm_active = True
                            pygame.mixer.music.play()
                            socketio.emit('drowsiness_alert', {'message': 'DROWSINESS ALERT!'})
                        cv2.putText(frame, "DROWSINESS ALERT!", (50, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                else:
                    frame_counter = max(0, frame_counter - 1)  # Gradually decrease counter
                    if frame_counter == 0:
                        with lock:
                            alarm_active = False
                            pygame.mixer.music.stop()
                
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    cap.release()

def start_video_feed():
    socketio.run(app, debug=True, use_reloader=False)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    threading.Thread(target=start_video_feed).start()
