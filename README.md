# Road Defect Indexing System

A comprehensive system for detecting, analyzing, and indexing road defects using computer vision and machine learning. This system provides real-time detection of road defects through a camera feed or video file, processes the defects using advanced image processing techniques, and calculates severity using fuzzy logic.

## Features

- Real-time road defect detection using YOLOv10
- Multiple defect type detection:
  - Linear Cracks
  - Alligator Cracks
  - Potholes
- Support for both live camera feed and video file analysis
- Advanced image processing for defect enhancement
- Fuzzy logic-based severity calculation
- GPS integration for defect location tracking
- Cloud storage integration
- Real-time visualization dashboard 
- Defect statistics and analysis
- Run analysis for detailed defect assessment (Coming Soon)
- Camera controls (zoom, flip)
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
1. Clone the repository with submodules:
```bash
# Method 1: Clone with submodules in one command
git clone --recurse-submodules https://github.com/yourusername/Road-Defect-Indexing-System.git

# OR Method 2: If you've already cloned without submodules
git clone https://github.com/yourusername/Road-Defect-Indexing-System.git
cd Road-Defect-Indexing-System
git submodule update --init --recursive
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

1. Clone the repository with submodules:
```bash
# Method 1: Clone with submodules in one command
git clone --recurse-submodules https://github.com/yourusername/Road-Defect-Indexing-System.git

# OR Method 2: If you've already cloned without submodules
git clone https://github.com/yourusername/Road-Defect-Indexing-System.git
cd Road-Defect-Indexing-System
git submodule update --init --recursive
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
- Place the model file (`road_defect.pt`) in the `models` directory
- The model should be compatible with YOLOv10

## Project Structure

```
Road-Defect-Indexing-System/
├── src/
│   ├── app.py                 # Main application entry point
│   ├── modules/
│   │   ├── camera.py          # Camera handling
│   │   ├── detection.py       # Defect detection
│   │   ├── gps_reader.py      # GPS integration
│   │   └── cloud_connector.py # Cloud storage
│   ├── ui/
│   │   ├── dashboard.py       # Main GUI dashboard
│   │   ├── video_controls.py  # Video/camera controls
│   │   ├── main_controls.py   # Main control panel
│   │   ├── statistics.py      # Statistics display
│   │   └── status_bar.py      # Status bar
│   └── models/
│       └── yolov10/           # YOLOv10 model submodule
├── public/
│   └── icons/                 # Application icons
├── config/
│   └── settings.json          # Application settings
├── requirements.txt
├── setup.bat
└── start.bat
```

## Usage

### Starting the Application

1. Ensure all dependencies are installed and the model is in place
2. Run the main application:
```bash
python -m src.app
```

### Using the Dashboard

1. **Video Source Selection**:
   - Choose between camera feed or video file
   - For camera: Select from available cameras
   - For video: Upload a video file

2. **Camera Controls** (when using camera):
   - Adjust zoom level
   - Flip camera view if needed

3. **Video Playback Controls** (when using video file):
   - Play/Pause
   - Rewind/Forward
   - Progress tracking

4. **Detection**:
   - Click "Start Detection" to begin defect detection
   - View detection results in real-time
   - Monitor defect statistics in the dashboard

5. **GPS Integration**:
   - Automatically connects to available GPS
   - GPS coordinates are logged with detected defects

6. **Cloud Integration**:
   - Connect to cloud storage
   - Upload detection data
   - Manage cloud storage

7. **Analysis** (Coming Soon):
   - Run detailed analysis on recorded defects
   - Generate comprehensive reports
   - View severity trends and patterns
   - Export analysis results

### Configuration

The system can be configured through the Settings dialog:
- Confidence threshold adjustment
- Output directory selection
- Recording output directory
- Record mode toggle
- Cloud storage settings
- Analysis settings 

## Technical Details

### Defect Detection
- Uses YOLOv10 for real-time object detection
- Configurable confidence threshold
- Supports multiple defect types
- Real-time processing

### Image Processing Pipeline
1. Image acquisition
2. Preprocessing
3. Defect detection
4. Post-processing
5. Severity calculation

### GPS Integration
- Supports NMEA-compatible GPS modules
- Automatic port detection
- Real-time coordinate logging

### Cloud Storage
- Supports cloud storage integration
- Efficient data transmission
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
