# Road Defect Indexing System

A comprehensive system for detecting, analyzing, and indexing road defects using computer vision, fuzzy logic and Random Forest. This system provides real-time detection of road defects through a camera feed, processes the defects using advanced image processing techniques, and calculates severity using fuzzy logic.

## Features

- Real-time road defect detection using YOLO
- Multiple defect type detection:
  - Linear Cracks
  - Alligator Cracks
  - Potholes
- Advanced image processing for defect enhancement
- Fuzzy logic-based severity calculation
- GPS integration for defect location tracking
- Cloud storage integration
- Real-time visualization dashboard 
- Defect statistics and Predictive Analysis using Random Forest 
- Camera controls (zoom, brightness, exposure)
- Support for multiple camera inputs

## System Requirements

### Hardware Requirements
- Camera (minimum 720p resolution, 1280x720 or Higher recommended)
- GPS module (optional, for location tracking)
- GPU recommended for real-time detection (NVIDIA with CUDA support)

### Software Requirements
- Python 3.8 or higher
- OpenCV 4.x
- PyQt5
- CUDA Toolkit (if using GPU)
- Other dependencies listed in requirements.txt

## Installation

### Easy Installation (Windows)
1. Clone the repository:
```bash
git clone https://github.com/yourusername/Road-Defect-Indexing-System.git
cd Road-Defect-Indexing-System
```

2. Run the setup script:
```bash
setup.bat
```
This will automatically:
- Check Python installation
- Create a virtual environment
- Install all required packages
- Check for CUDA support
- Create necessary directories
- Verify YOLO model presence

3. Start the application:
```bash
start.bat
```

### Manual Installation
If you prefer to install manually or are using a different operating system:

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Road-Defect-Indexing-System.git
cd Road-Defect-Indexing-System
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Download the YOLO model:
- Place the model file (`road_defect_v2.pt`) in the `models` directory
- The model should be compatible with YOLOv10

## Project Structure

```
Road-Defect-Indexing-System/
├── src/
│   ├── app/
│   │   ├── components/
│   │   │   ├── dashboard.py      # Main GUI dashboard
│   │   │   └── ...
│   │   ├── modules/
│   │   │   ├── camera.py         # Camera handling
│   │   │   ├── detection.py      # Defect detection
│   │   │   ├── fuzzy_logic.py    # Fuzzy logic calculations
│   │   │   ├── severity_calculator.py  # Defect analysis
│   │   │   ├── gps_reader.py     # GPS integration
│   │   │   └── cloud_connector.py # Cloud storage
│   │   └── ...
│   └── ...
├── models/
│   └── road_defect_v2.pt         # YOLO model
├── requirements.txt
└── README.md
```

## Usage

### Starting the Application

1. Ensure all dependencies are installed and the model is in place
2. Run the main application:
```bash
python -m src.app.main
```

### Using the Dashboard

1. **Camera Setup**:
   - Select camera from dropdown menu
   - Adjust camera settings (zoom, brightness, exposure)
   - Use "Flip Camera" button if needed

2. **Detection**:
   - Click "Start Detection" to begin real-time defect detection
   - View detection results in real-time
   - Monitor defect statistics in the dashboard

3. **GPS Integration**:
   - Click "Connect GPS" to enable location tracking
   - GPS coordinates will be logged with detected defects

4. **Analysis**:
   - Click "Run Analysis" for detailed defect analysis
   - View severity calculations and statistics

5. **Cloud Integration**:
   - Click "Connect Cloud" to enable cloud storage
   - Use "Upload Data" to send detection data to cloud
   - Use "Send Data" for real-time data transmission

### Configuration

The system can be configured through the Settings dialog:
- Confidence threshold adjustment
- Output directory selection
- Camera parameters
- Processing parameters

## Technical Details

### Defect Detection
- Uses YOLOv10 for real-time object detection
- Minimum confidence threshold: 0.25
- Supports multiple defect types
- Real-time processing at 720p resolution

### Image Processing Pipeline
1. Image acquisition
2. Preprocessing:
   - NLM denoising
   - Contrast enhancement
   - Bilateral filtering
3. Defect detection
4. Post-processing:
   - Crack enhancement
   - Noise reduction
   - Morphological operations
   - Super Pixel Segementation

### Severity Calculation
Uses fuzzy logic to calculate defect severity based on:
- Defect dimensions
- Defect ratio
- Vehicle damage risk
- Traffic impact
- Defect type

### GPS Integration
- Supports NMEA-compatible GPS modules
- Automatic port detection
- Real-time coordinate logging

### Cloud Storage
- Supports cloud storage integration
- Efficient Data transmission
- Defect metadata storage
- Image backup

## Troubleshooting

### Common Issues

1. **Camera Not Detected**:
   - Check camera connection
   - Verify camera permissions
   - Try different camera index

2. **Low Detection Accuracy**:
   - Adjust confidence threshold
   - Check lighting conditions
   - Verify camera focus

3. **GPS Connection Issues**:
   - Check GPS module connection
   - Verify correct COM port
   - Ensure clear sky view

4. **Performance Issues**:
   - Enable GPU acceleration
   - Reduce processing resolution
   - Close unnecessary applications

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLO team for the object detection framework
- OpenCV community
- PyQt team
- Contributors and maintainers

## Contact

For support or queries, please open an issue in the repository or contact the maintainers.

## Version History

- v1.0.0: Initial release
  - Basic defect detection
  - Dashboard implementation
  - GPS integration
  - Cloud storage support 
