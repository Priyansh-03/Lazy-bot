import cv2
from pygrabber.dshow_graph import FilterGraph

class CameraSelector:
    @staticmethod
    def list_available_cameras():
        graph = FilterGraph()
        device_list = graph.get_input_devices()
        return device_list

    @staticmethod
    def select_camera():
        available_cameras = CameraSelector.list_available_cameras()

        if not available_cameras:
            print("⚠️ Koi camera nahi mila! Bandh karo...")
            exit()

        print("\nAvailable Cameras:")
        for idx, cam in enumerate(available_cameras):
            print(f"🔹 [{idx}] {cam}")

        while True:
            try:
                camera_index = int(input("\nCamera index daalo: "))
                
                if 0 <= camera_index < len(available_cameras):
                    cap = cv2.VideoCapture(camera_index)
                    if cap.isOpened():
                        cap.release()
                        return camera_index
                    else:
                        print("⚠️ Yeh camera nahi chal raha. Dusra chuno.")
                else:
                    print("⚠️ Galat index hai. List se chuno.")

            except ValueError:
                print("⚠️ Number daalo bhai!")
