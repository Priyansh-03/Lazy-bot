import os
import json
import math

def preprocess_annotations():
    postures = ["straight", "bent_forward", "bent_backward", "sideways", "lazy_pose", "yawn", "smile", "neutral"]
    processed_data = []

    for posture in postures:
        annotation_dir = os.path.join("D:/Programs/Projects/Lazy bot/data/annotations", posture)
        for annotation_file in os.listdir(annotation_dir):
            if annotation_file.endswith(".json"):
                with open(os.path.join(annotation_dir, annotation_file), 'r') as f:
                    data = json.load(f)

                for frame_data in data:
                    frame_path = frame_data["frame"].replace("\\", "/")
                    pose_landmarks = frame_data["pose_landmarks"]
                    if len(pose_landmarks) > mp_pose.PoseLandmark.RIGHT_SHOULDER.value:
                        nose = pose_landmarks[mp_pose.PoseLandmark.NOSE.value]
                        left_shoulder = pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                        right_shoulder = pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                        
                        # Calculate chest position
                        chest_x = (left_shoulder[0] + right_shoulder[0]) / 2
                        chest_y = (left_shoulder[1] + right_shoulder[1]) / 2 + 0.1
                        nose_pos = [nose[0], nose[1]]
                        chest_pos = [chest_x, chest_y]

                        # Calculate spine angle (in degrees)
                        delta_y = chest_pos[1] - nose_pos[1]
                        delta_x = chest_pos[0] - nose_pos[0]
                        spine_angle = math.degrees(math.atan2(delta_y, delta_x))
                        spine_angle = abs(spine_angle - 90)  # Deviation from vertical (90°)

                        # Ensure angle is between 0° and 90°
                        spine_angle = min(max(spine_angle, 0), 90)

                        processed_data.append({
                            "frame": frame_path,
                            "posture": posture,
                            "spine_angle": spine_angle,
                            "label": 0 if posture == "straight" else 1
                        })

    # Save processed annotations
    with open("D:/Programs/Projects/Lazy bot/data/annotations/processed_data.json", "w") as f:
        json.dump(processed_data, f, indent=4)

if __name__ == "__main__":
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    preprocess_annotations()
    print("Processed annotations saved to D:/Programs/Projects/Lazy bot/data/annotations/processed_data.json")