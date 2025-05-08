import cv2
import numpy as np
from ultralytics import YOLOv10 as YOLO
from openvino.runtime import Core
import torch

class OpenVINOModel:
    def __init__(self, model_path='src/app/models/last.pt'):
        # Initialize YOLO model
        self.model = YOLO(model_path)
        
        # Export to OpenVINO format
        self.model.export(format="openvino", imgsz=640)
        
        # Initialize OpenVINO runtime
        self.ie = Core()
        
        # Load the OpenVINO model
        self.ov_model = self.ie.read_model(f"{model_path[:-3]}_openvino_model/model.xml")
        self.compiled_model = self.ie.compile_model(model=self.ov_model, device_name="CPU")
        
        # Get input and output layers
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)
        
    def preprocess_image(self, image):
        # Resize image to model input size
        resized = cv2.resize(image, (640, 640))
        
        # Convert to RGB if needed
        if len(resized.shape) == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        elif resized.shape[2] == 4:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGRA2RGB)
        elif resized.shape[2] == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
        # Normalize and transpose
        input_tensor = np.expand_dims(resized.transpose(2, 0, 1), 0).astype(np.float32) / 255.0
        
        return input_tensor
    
    def predict(self, image):
        # Preprocess image
        input_tensor = self.preprocess_image(image)
        
        # Run inference
        results = self.compiled_model([input_tensor])[self.output_layer]
        
        # Process results
        predictions = self.model.postprocess(torch.from_numpy(results))
        
        return predictions[0]  # Return first batch item
    
    def __call__(self, image):
        return self.predict(image) 