import tkinter as tk
from PIL import Image, ImageTk
import cv2

class UI:
    def __init__(self, root):
        self.root = root
        self.root.title("LazyBot Posture Guide")
        self.feedback_text = []

        # Left frame for camera feed
        self.left_frame = tk.Frame(self.root, width=800, height=600, bd=2, relief="solid")
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Right panel for feedback
        self.right_panel = tk.Canvas(self.root, width=400, height=600, bg="black")
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH)

        self.label = tk.Label(self.left_frame)
        self.label.pack()

        self.root.update()

    def update_frame(self, frame=None, feedback=None):
        if frame is not None:
            try:
                frame = cv2.resize(frame, (800, 600))
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                self.label.imgtk = imgtk
                self.label.configure(image=imgtk)
            except Exception as e:
                print(f"Error updating frame: {e}")

        if feedback is not None:
            self.feedback_text = feedback

        # Clear and update right panel
        self.right_panel.delete("all")
        self.right_panel.create_text(200, 30, text="Posture Guide", fill="white", font=("Helvetica", 12, "bold"))

        # Display feedback with color coding
        for i, text in enumerate(self.feedback_text):
            color = "green" if "✅" in text or "Straight" in text else "red" if "⚠️" in text or "Bent" in text else "white"
            self.right_panel.create_text(200, 60 + i * 25, text=text, fill=color, font=("Helvetica", 10))

        self.root.update()