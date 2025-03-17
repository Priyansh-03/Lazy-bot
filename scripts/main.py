import cv2
import time
import numpy as np
from pynput.keyboard import Listener as keyboardlistener
from pynput.mouse import Listener as mouseListener
from camera_selector import CameraSelector
from pose_estimation import PoseEstimator
from ui import UI
import tkinter as tk

class LazyBot:
    def __init__(self, camera_index):
        self.last_activity = time.time()
        self.cap = cv2.VideoCapture(camera_index)
        self.pose_estimator = PoseEstimator()
        self.feedback_text = []

        if not self.cap.isOpened():
            print("⚠️ Error: Camera nahi chal raha!")
            self.cap.release()
            raise Exception("Camera nahi chal raha!")

        # Initialize frames for movement detection
        ret, self.frame1 = self.cap.read()
        ret, self.frame2 = self.cap.read()
        if not ret:
            print("⚠️ Error: Initial frames not captured!")
            self.cap.release()
            raise Exception("Initial frames not captured!")

    def on_key_press(self, key):
        self.last_activity = time.time()

    def on_mouse_click(self, x, y, button, pressed):
        if pressed:
            self.last_activity = time.time()

    def reset_posture(self):
        self.feedback_text = ["✅ Posture reset ho gaya!"]
        self.pose_estimator.reset()
        self.last_activity = time.time()

    def monitor_activity(self):
        keyboard_listener = keyboardlistener(on_press=self.on_key_press)
        mouse_listener = mouseListener(on_click=self.on_mouse_click)
        keyboard_listener.start()
        mouse_listener.start()

        self.ui = UI(tk.Tk())
        good_frames = 0
        bad_frames = 0
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        if fps == 0:  # Avoid division by zero
            fps = 30
        last_warning = time.time()

        def update_loop():
            nonlocal good_frames, bad_frames, last_warning
            ret, frame = self.cap.read()
            if not ret:
                print("⚠️ Error: Frame not captured!")
                self.ui.root.quit()
                return

            if time.time() - self.last_activity > 0.05:
                # Process frame for posture detection
                frame = self.pose_estimator.estimate_pose(frame)
                self.feedback_text = self.pose_estimator.get_feedback(good_frames, bad_frames, fps)

                # Movement detection
                gray1 = cv2.cvtColor(self.frame1, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(self.frame2, cv2.COLOR_BGR2GRAY)
                diff = cv2.absdiff(gray1, gray2)
                blur = cv2.GaussianBlur(diff, (5, 5), 0)
                _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
                movement = np.sum(thresh)

                self.frame1 = self.frame2
                self.frame2 = frame

                # Update posture counters based on movement and posture
                if movement < 5000:
                    if self.pose_estimator.is_bad_posture(self.pose_estimator.smooth_spine_angle):
                        good_frames = 0
                        bad_frames += 1
                        if (time.time() - last_warning > 5) and (bad_frames / fps > 180):
                            self.feedback_text.append("⚠️ Warning: Bad posture detected for too long!")
                            last_warning = time.time()
                    else:
                        bad_frames = 0
                        good_frames += 1
                else:
                    good_frames += 1
                    bad_frames = 0

                # Update UI with frame and feedback
                self.ui.update_frame(frame, self.feedback_text)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.ui.root.quit()
                return

            self.ui.root.after(10, update_loop)  # Schedule next update

        update_loop()
        self.ui.root.mainloop()

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    selected_camera = None
    try:
        selected_camera = CameraSelector.select_camera()
        if selected_camera is not None:
            bot = LazyBot(selected_camera)
            bot.monitor_activity()
    except Exception as e:
        print(f"⚠️ Error: {e}")
        if selected_camera is not None:
            print("🔄 Camera issue detected with selected index. Program exiting...")
        else:
            print("🔄 No camera selected or all attempts failed. Program exiting...")