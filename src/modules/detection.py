import cv2
import torch
import numpy as np
from ultralytics import YOLOv10 as YOLO
import os
import json
import logging
from datetime import datetime
from PIL import Image
import threading
import queue
import piexif
from concurrent.futures import ThreadPoolExecutor
import time
from typing import Optional, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DefectDetector:
    def __init__(self, model_path, save_dir=None, gps_reader=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info(f"Using device: {self.device}")

        self.model = YOLO(model_path).to(self.device)
        self.model.conf = 0.25
        self.model.iou = 0.45
        self.model.max_det = 50
        self.model.agnostic = True
        self.model.multi_label = False
        self.model.verbose = False

        self.class_names = ['Linear-Crack', 'Alligator-Crack', 'Pothole']
        self.save_dir = save_dir or os.path.join(os.path.expanduser("~"), "Raw Detections")
        os.makedirs(self.save_dir, exist_ok=True)

        self.confidence_threshold = 0.25
        self.gps_reader = gps_reader
        self.last_gps_update = 0
        self.gps_update_interval = 1.0

        self.defect_colors = {
            'Linear-Crack': (255, 105, 180),
            'Alligator-Crack': (128, 0, 128),
            'Pothole': (0, 0, 255),
        }

        self.frame_counts = {name: 0 for name in self.class_names}
        self.defect_counts = {}
        self.save_queue = queue.Queue()
        self.save_thread = threading.Thread(target=self._save_worker, daemon=True)
        self.save_thread.start()

        self.thread_pool = ThreadPoolExecutor(max_workers=os.cpu_count())

    def _save_worker(self):
        while True:
            frame, metadata = self.save_queue.get()
            if frame is None:
                break
            self._save_detection(frame, metadata)
            self.save_queue.task_done()

    def _convert_to_exif_gps(self, coord):
        degrees = int(coord)
        minutes_float = abs((coord - degrees) * 60)
        minutes = int(minutes_float)
        seconds = int((minutes_float - minutes) * 60 * 10000)
        return ((abs(degrees), 1), (minutes, 1), (seconds, 10000))

    def _get_valid_gps_data(self) -> Tuple[Optional[float], Optional[float]]:
        """Get GPS data from the GPS reader"""
        if not self.gps_reader:
            return None, None
        
        try:
            lat, lon = self.gps_reader.read_gps_data()
            if lat is not None and lon is not None:
                return lat, lon
        except Exception as e:
            logging.error(f"GPS read error: {e}")
        return None, None

    def _save_detection(self, frame, metadata):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_filename = f"detection_{timestamp}.jpg"
            image_path = os.path.join(self.save_dir, image_filename)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)

            location = metadata.get("Location", {})
            lat = location.get("latitude")
            lon = location.get("longitude")

            exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}

            if lat is not None and lon is not None and -90 <= lat <= 90 and -180 <= lon <= 180:
                lat_exif = self._convert_to_exif_gps(abs(lat))
                lon_exif = self._convert_to_exif_gps(abs(lon))

                exif_dict['GPS'][piexif.GPSIFD.GPSLatitudeRef] = 'N' if lat >= 0 else 'S'
                exif_dict['GPS'][piexif.GPSIFD.GPSLatitude] = lat_exif
                exif_dict['GPS'][piexif.GPSIFD.GPSLongitudeRef] = 'E' if lon >= 0 else 'W'
                exif_dict['GPS'][piexif.GPSIFD.GPSLongitude] = lon_exif
                exif_dict['GPS'][piexif.GPSIFD.GPSVersionID] = (2, 3, 0, 0)

            try:
                minimal_metadata = {
                    "Location": {
                        "latitude": float(lat) if lat is not None else None,
                        "longitude": float(lon) if lon is not None else None,
                        "timestamp": location.get("timestamp") if location else None
                    }
                }
                comment_str = json.dumps(minimal_metadata)
                encoded_comment = b"ASCII\0\0\0" + comment_str.encode('utf-8')
                exif_dict['Exif'][piexif.ExifIFD.UserComment] = encoded_comment
            except Exception as e:
                logging.error(f"Encoding location metadata failed: {e}")

            try:
                exif_bytes = piexif.dump(exif_dict)
                pil_image.save(image_path, "jpeg", exif=exif_bytes)
                logging.info(f"Image saved with EXIF: {image_path}")
            except Exception as e:
                logging.error(f"Saving image with EXIF failed: {e}")
                pil_image.save(image_path, "jpeg")
                logging.info(f"Image saved without EXIF: {image_path}")
        except Exception as e:
            logging.error(f"Error in save_detection: {e}")

    def detect(self, frame):
        frame_with_boxes = frame.copy()
        for name in self.class_names:
            self.frame_counts[name] = 0
        self.defect_counts.clear()

        try:
            lat, lon = self._get_valid_gps_data()

            h, w = frame.shape[:2]
            input_size = (640, 640)
            scale_x = w / input_size[0]
            scale_y = h / input_size[1]

            resized = cv2.resize(frame, input_size)

            results = self.model(resized,
                                 conf=self.confidence_threshold,
                                 verbose=False,
                                 device=self.device,
                                 agnostic_nms=True,
                                 max_det=50)

            if results[0].boxes and len(results[0].boxes) > 0:
                boxes = results[0].boxes.cpu()
                location_metadata = {
                    "Location": {
                        "latitude": lat,
                        "longitude": lon,
                        "timestamp": datetime.now().isoformat()
                    }
                } if lat is not None and lon is not None else None

                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x1, x2 = x1 * scale_x, x2 * scale_x
                    y1, y2 = y1 * scale_y, y2 * scale_y

                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = self.class_names[class_id]

                    color = self.defect_colors.get(class_name, (255, 0, 0))
                    cv2.rectangle(frame_with_boxes, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    label = f"{class_name} {confidence:.2f}"
                    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(frame_with_boxes, (int(x1), int(y1) - text_h - 10), (int(x1) + text_w, int(y1)), color, -1)
                    cv2.putText(frame_with_boxes, label, (int(x1), int(y1) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    self.defect_counts[class_name] = self.defect_counts.get(class_name, 0) + 1
                    self.frame_counts[class_name] += 1

                if location_metadata:
                    self.save_queue.put((frame, location_metadata))
                    logging.info(f"Queued image with GPS metadata: {location_metadata['Location']}")
                else:
                    logging.warning("Skipping save - No valid GPS data")

            return frame_with_boxes, self.frame_counts.copy()

        except Exception as e:
            logging.error(f"Error during detection: {e}")
            return frame_with_boxes, {name: 0 for name in self.class_names}

    def cleanup(self):
        self.save_queue.put((None, None))
        self.save_thread.join()
        self.thread_pool.shutdown(wait=True)



if __name__ == "__main__":
    from camera import Camera

    class MockGPSReader:
        def get_gps_data(self):
            # Simulated GPS coordinates (e.g., San Francisco)
            return 37.7749, -122.4194

    # Replace with the actual path to your YOLOv10 model
    model_path = "C:/Users/bentl/Desktop/RoadDefectSystem/src/models/road_defect.pt"

    # Initialize the detector
    detector = DefectDetector(model_path=model_path, gps_reader=MockGPSReader())

    try:
        # Initialize your camera
        camera = Camera()
        print(f"Camera initialized with resolution: {camera.max_resolution} at {camera.max_fps} FPS")
        camera.set_zoom(1.0)
        camera.set_flipped(vertical=False, horizontal=False)

        print("Press 'q' to exit the test loop.")

        while True:
            ret, frame = camera.capture.read()
            if not ret:
                print("Failed to read frame from camera.")
                break

            # Apply digital zoom and flipping if set
            frame = camera.digital_zoom(frame, camera.zoom_factor)
            frame = camera.flip_frame(frame, camera.flip_vertical, camera.flip_horizontal)

            # Run detection
            processed_frame, counts = detector.detect(frame)

            # Display the frame
            cv2.imshow("Detections", processed_frame)
            print("Defect counts:", counts)

            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Exception occurred: {e}")

    finally:
        detector.cleanup()
        camera.cleanup()
