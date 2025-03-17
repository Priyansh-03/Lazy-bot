import cv2
import os
import time
import json
import numpy as np
from collections import deque
import mediapipe as mp

# Base Paths
BASE_PATH = "./data"
VIDEO_PATH = os.path.join(BASE_PATH, "videos")
FRAME_PATH = os.path.join(BASE_PATH, "frames")
ANNOTATION_PATH = os.path.join(BASE_PATH, "annotations")
os.makedirs(VIDEO_PATH, exist_ok=True)
os.makedirs(FRAME_PATH, exist_ok=True)
os.makedirs(ANNOTATION_PATH, exist_ok=True)

# Mediapipe Setup
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()
face_mesh = mp_face.FaceMesh()

# Skeleton Mapping Function
def get_skeleton_landmarks(pose_landmarks):
    return [[lm.x, lm.y, lm.z] for lm in pose_landmarks.landmark]

# Facial Expression Extraction
def get_facial_expressions(face_landmarks):
    key_points = [33, 263, 13, 14, 61, 291, 199]  # Nose, Eyes, Lips key points
    if not face_landmarks:
        return {}
    return {idx: [face_landmarks.landmark[idx].x, face_landmarks.landmark[idx].y] for idx in key_points if idx < len(face_landmarks.landmark)}

# Pixel Change Calculation
def calculate_pixel_change(prev_frame, curr_frame):
    if prev_frame is None:
        return 0
    diff = cv2.absdiff(prev_frame, curr_frame)
    return np.sum(diff) / (diff.shape[0] * diff.shape[1])

# Record Postures with Skeleton Mapping & Facial Expressions
def record_postures():
    cap = cv2.VideoCapture(0)
    postures = ["straight", "bent_forward", "bent_backward", "sideways", "lazy_pose", "yawn", "smile", "neutral"]
    transition_time = 2
    record_time = 10
    prev_gray = None

    for posture in postures:
        posture_video_path = os.path.join(VIDEO_PATH, posture)
        posture_frame_path = os.path.join(FRAME_PATH, posture)
        posture_annotation_path = os.path.join(ANNOTATION_PATH, posture)
        os.makedirs(posture_video_path, exist_ok=True)
        os.makedirs(posture_frame_path, exist_ok=True)
        os.makedirs(posture_annotation_path, exist_ok=True)

        video_filename = os.path.join(posture_video_path, f"{posture}_{int(time.time())}.mp4")
        video_writer = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (640, 480))
        start_time = time.time()
        annotation_data = []
        frame_count = 0

        while time.time() - start_time < record_time:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            pixel_change = calculate_pixel_change(prev_gray, gray)
            prev_gray = gray
            
            # Save frame
            frame_filename = os.path.join(posture_frame_path, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_filename, frame)
            
            # Pose Estimation
            results_pose = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results_pose.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            pose_data = get_skeleton_landmarks(results_pose.pose_landmarks) if results_pose.pose_landmarks else []
            
            # Face Expression Analysis
            results_face = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            face_data = {}
            if results_face.multi_face_landmarks:
                face_data = get_facial_expressions(results_face.multi_face_landmarks[0])
                for face_landmarks in results_face.multi_face_landmarks:
                    mp_drawing.draw_landmarks(frame, face_landmarks, landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1))
            
            annotation_data.append({
                "timestamp": time.time(),
                "frame": frame_filename,
                "pose_landmarks": pose_data,
                "facial_expressions": face_data,
                "pixel_change": pixel_change
            })
            
            video_writer.write(frame)
            cv2.putText(frame, f"{posture}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Recording", frame)
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Save Annotation Data
        annotation_filename = os.path.join(posture_annotation_path, f"{posture}_{int(time.time())}.json")
        with open(annotation_filename, 'w') as f:
            json.dump(annotation_data, f, indent=4)
        
        video_writer.release()
        print(f"Recorded {posture}, transitioning...")
        time.sleep(transition_time)
    
    cap.release()
    cv2.destroyAllWindows()
    print("Recording Complete!")

if __name__ == "__main__":
    record_postures()
