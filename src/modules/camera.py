import cv2
import numpy as np
import time
import atexit
import signal

try:
    import comtypes
    import comtypes.client
    DIRECTSHOW_AVAILABLE = True
except ImportError:
    DIRECTSHOW_AVAILABLE = False


class Camera:
    def __init__(self, camera_index=None):
        self.camera_index = camera_index if camera_index is not None else 0
        self.capture = cv2.VideoCapture(self.camera_index)

        if not self.capture.isOpened():
            print(f"Failed to open camera {self.camera_index}. Camera will be disabled.")  # Debug print
            self.is_available = False  # Set flag to indicate camera is not available
            return  # Exit the constructor if the camera cannot be opened

        self.is_available = True  # Camera is available
        self.max_fps = self.get_max_fps()
        self.set_max_resolution()

        self.zoom_factor = 1.0
        self.flip_vertical = False
        self.flip_horizontal = False

        atexit.register(self.cleanup)
        signal.signal(signal.SIGINT, self.signal_handler)

    def get_max_fps(self):
        common_fps = [30, 60, 120]
        supported_fps = []
        for fps in common_fps:
            self.capture.set(cv2.CAP_PROP_FPS, fps)
            actual_fps = int(self.capture.get(cv2.CAP_PROP_FPS))
            if actual_fps == fps:
                supported_fps.append(actual_fps)
        return max(supported_fps) if supported_fps else 30

    def set_max_resolution(self):
        # Try common high resolutions from highest to lower
        candidates = [
            (3840, 2160),  # 4K UHD
            (2560, 1440),  # QHD
            (1920, 1080),  # FHD
            (1280, 720),   # HD
        ]

        for width, height in candidates:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

            actual_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if (actual_width, actual_height) == (width, height):
                self.max_resolution = (width, height)
                print(f"Set resolution to {width}x{height}")
                return

        # If none matched, fall back to current resolution
        w = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.max_resolution = (w, h)
        print(f"Using fallback resolution {w}x{h}")

    def set_zoom(self, zoom_factor):
        if zoom_factor < 1.0:
            zoom_factor = 1.0
        self.zoom_factor = zoom_factor

    def set_flipped(self, vertical=False, horizontal=False):
        self.flip_vertical = vertical
        self.flip_horizontal = horizontal

    def flip_frame(self, frame, flip_vertical=False, flip_horizontal=False):
        if flip_vertical:
            frame = cv2.flip(frame, 0)
        if flip_horizontal:
            frame = cv2.flip(frame, 1)
        return frame

    @staticmethod
    def digital_zoom(frame, zoom_factor):
        if zoom_factor <= 1.0:
            return frame

        h, w = frame.shape[:2]
        new_w, new_h = int(w / zoom_factor), int(h / zoom_factor)

        x1 = (w - new_w) // 2
        y1 = (h - new_h) // 2

        zoomed_frame = frame[y1:y1+new_h, x1:x1+new_w]
        return cv2.resize(zoomed_frame, (w, h), interpolation=cv2.INTER_LINEAR)

    def cleanup(self):
        if self.is_available:
            print("Cleaning up camera resources...")
            self.capture.release()
            cv2.destroyAllWindows()
            print("Cleanup done.")

    def signal_handler(self, signum, frame):
        print("Termination signal received. Cleaning up...")
        self.cleanup()
        exit(0)
        


if __name__ == "__main__":
    cam = Camera()  # create camera instance
    
    print(f"Camera initialized with resolution: {cam.max_resolution} at {cam.max_fps} FPS")
    cam.set_zoom(1.0)
    cam.set_flipped(vertical=False, horizontal=False)

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cam.capture.read()
            if not ret:
                print("Failed to grab frame")
                break

            frame = cam.digital_zoom(frame, cam.zoom_factor)
            frame = cam.flip_frame(frame, cam.flip_vertical, cam.flip_horizontal)

            cv2.imshow('Live Camera Test', frame)

            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 1.0:
                print(f"FPS: {frame_count / elapsed:.2f}")
                frame_count = 0
                start_time = time.time()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exit requested")
                break

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        cam.cleanup()


