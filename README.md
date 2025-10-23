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
- Interactive Web Mapping Interface  
  - Displays detected road defects with precise GPS coordinates on an interactive map.
  - [View Mapping Module →](https://github.com/rvnztolentino/road-defect-indexing-maps)

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
|       └── model.pt/          # your YOLOv10 model 
|
├── public/
│   └── icons/                 # Application icons
├── scripts/
│   └── testscripts/           # Test Scripts for Unit Testing or Feature Testing
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

## Related Repositories

- [Road Defect Mapping Web App](https://github.com/rvnztolentino/road-defect-indexing-maps)  
  Interactive Mapbox-based dashboard for visualizing YOLOv10-detected road defects with GPS precision and road-type filtering.  
  Developed as part of the **Comprehensive Road Defect Indexing System**.

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

## Building the Application

### Building into an Executable

The application can be built into a standalone executable using PyInstaller. This process packages all dependencies and resources into a single file.

#### Prerequisites
- Python 3.8 or higher
- PyInstaller (`pip install pyinstaller`)
- Pillow (`pip install pillow`)

#### Building Steps

1. Ensure you have all dependencies installed:
```bash
pip install -r requirements.txt
pip install pyinstaller pillow
```

2. Run the build script:
```bash
python build.py
```

The build script will:
- Convert the application icon to ICO format
- Package all necessary files and dependencies
- Create a single executable in the `dist` directory

#### Build Output
- The executable will be created in the `dist` directory
- All required resources (models, icons, etc.) will be included
- The application will run without requiring Python installation

#### Notes
- The build process excludes PyQt5 to prevent conflicts with PyQt6
- The executable includes all necessary dependencies
- The build process may take several minutes to complete
- The resulting executable is self-contained and can be distributed 
