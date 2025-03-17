import cv2
import mediapipe as mp
import numpy as np
import math
import tensorflow as tf
import pickle

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class PoseEstimator:
    def __init__(self):
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.feedback = []
        self.smooth_spine_angle = None
        self.alpha = 0.3
        # Load trained model and scaler
        self.model = tf.keras.models.load_model("D:/Programs/Projects/Lazy bot/models/trained/posture_model.h5")
        with open("D:/Programs/Projects/Lazy bot/models/trained/scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)

    def reset(self):
        self.feedback = ["✅ Posture reset ho gaya!"]
        self.smooth_spine_angle = None

    def smooth_value(self, current, previous):
        return current if previous is None else self.alpha * current + (1 - self.alpha) * previous

    def is_bad_posture(self, predicted_label=None):
        if predicted_label is None:
            return False
        return predicted_label > 0.5  # Threshold adjusted back to 0.5

    def get_feedback(self, good_frames, bad_frames, fps):
        self.feedback = [
            "Skeleton landmarks used for posture detection:",
            "Blue lines: Body skeleton",
            "Red dots: Key joints"
        ]
        if good_frames > 0:
            self.feedback.append(f"✅ Good Posture Time: {round((1/fps)*good_frames, 1)}s")
        if bad_frames > 0:
            self.feedback.append(f"⚠️ Bad Posture Time: {round((1/fps)*bad_frames, 1)}s")
        posture_status = "Straight" if not self.is_bad_posture(self.smooth_spine_angle) else "Bent"
        self.feedback.append(f"Current Posture: {posture_status}")
        return self.feedback

    def estimate_pose(self, frame):
        try:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image)
        except Exception as e:
            print(f"Pose processing error: {e}")
            return frame

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            h, w = frame.shape[:2]

            # Draw full skeleton using Mediapipe drawing utils
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
            )

            # Extract key landmarks for spine angle calculation
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

            # Calculate spine angle using shoulders and hips
            shoulder_mid_x = (left_shoulder.x + right_shoulder.x) / 2
            shoulder_mid_y = (left_shoulder.y + right_shoulder.y) / 2
            hip_mid_x = (left_hip.x + right_hip.x) / 2
            hip_mid_y = (left_hip.y + right_hip.y) / 2

            # Calculate spine angle (shoulder midpoint to hip midpoint deviation from vertical)
            delta_y = hip_mid_y - shoulder_mid_y
            delta_x = hip_mid_x - shoulder_mid_x
            raw_spine_angle = math.degrees(math.atan2(delta_y, delta_x))
            spine_angle = abs(raw_spine_angle - 90)  # Deviation from vertical (90°)

            # Adjust spine angle range for better differentiation
            if spine_angle < 10:
                spine_angle = spine_angle * 1.2  # Small amplification for straight posture
            elif spine_angle < 20:
                spine_angle = spine_angle * 2.0  # Medium amplification for slight bend
            else:
                spine_angle = spine_angle * 3.0  # High amplification for bent posture
            spine_angle = min(max(spine_angle, 0), 90)

            # Debug: Print raw and adjusted spine angle
            print(f"Raw Spine Angle: {raw_spine_angle:.3f}, Adjusted Spine Angle: {spine_angle:.3f}")

            # Normalize spine angle for prediction
            input_data = self.scaler.transform(np.array([[spine_angle]]))
            print(f"Normalized Input: {input_data[0][0]:.3f}")  # Debug normalized input
            predicted_label = self.model.predict(input_data, verbose=0)[0][0]
            self.smooth_spine_angle = self.smooth_value(predicted_label, self.smooth_spine_angle)

            # Debug: Print predicted label and smoothed value
            print(f"Predicted Label: {predicted_label:.3f}, Smooth Spine Angle: {self.smooth_spine_angle:.3f}")

            # Display predicted posture on frame with correct angle
            posture_text = "Straight" if predicted_label < 0.5 else "Bent"
            cv2.putText(frame, f"Posture: {posture_text} (Angle: {int(spine_angle)}°)", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                       (0, 255, 0) if predicted_label < 0.5 else (0, 0, 255), 2)

        return frame