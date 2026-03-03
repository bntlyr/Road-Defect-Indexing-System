# Road Defect Indexing System

A comprehensive system for detecting, analyzing, and indexing road defects using YOLOv10 computer vision and fuzzy logic severity assessment.

## Features

- **Real-time Detection** - YOLOv10-powered defect detection via camera or video file
- **Multi-Defect Support** - Linear cracks, alligator cracks, and potholes
- **GPS Integration** - Precise location tracking for all detected defects
- **Cloud Storage** - Automated data backup and synchronization
- **Live Dashboard** - Real-time visualization with statistics and analysis
- **Camera Controls** - Zoom, flip, and multi-camera support
- **Web Mapping** - [Interactive defect visualization](https://github.com/rvnztolentino/road-defect-indexing-maps)

## Requirements

### Hardware
- Camera: 720p minimum (1280x720+ recommended)
- GPS module: NMEA-compatible (required)
- GPU: NVIDIA with CUDA support (recommended)

### Software
- Python 3.8+
- OpenCV 4.x
- PyQt5
- CUDA Toolkit (for GPU acceleration)
- Dependencies in `requirements.txt`

## Quick Start

### Windows Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/Road-Defect-Indexing-System.git
cd Road-Defect-Indexing-System
```

2. **Run automated setup**
```bash
setup.bat
```
Automatically handles: Python verification, virtual environment creation, package installation, CUDA detection, directory setup, and model verification.

3. **Launch application**
```bash
start.bat
```

### Manual Start
```bash
python -m src.app
```

## Project Structure
```
Road-Defect-Indexing-System/
├── src/
│   ├── app.py                    # Application entry point
│   ├── modules/
│   │   ├── camera.py             # Camera handling
│   │   ├── detection.py          # Defect detection engine
│   │   ├── gps_reader.py         # GPS integration
│   │   └── cloud_connector.py    # Cloud storage
│   ├── ui/                       # GUI components
│   │   ├── dashboard.py
│   │   ├── video_controls.py
│   │   ├── main_controls.py
│   │   ├── statistics.py
│   │   └── status_bar.py
│   └── models/
│       ├── yolov10/              # YOLOv10 submodule
│       └── model.pt              # Trained model
├── public/icons/
├── scripts/testscripts/
├── requirements.txt
├── setup.bat
└── start.bat
```

## Usage Guide

### Dashboard Overview

**1. Video Source**
- Camera: Select from available devices
- Video File: Upload for batch processing

**2. Camera Controls** (Live mode)
- Zoom adjustment
- View orientation flip

**3. Playback Controls** (Video mode)
- Play/Pause, Rewind/Forward
- Progress tracking

**4. Detection**
- Start/Stop detection
- Real-time results visualization
- Live statistics monitoring

**5. GPS & Cloud**
- Auto-connects to GPS module
- Coordinates logged per defect
- Cloud sync for data backup

**6. Analysis** *(Coming Soon)*
- Detailed defect reports
- Severity trend analysis
- Export capabilities

### Configuration
Access via Settings dialog:
- Output directories
- Recording options
- Cloud storage credentials
- Analysis parameters

## Technical Overview

**Detection Pipeline**
1. Image acquisition
2. Preprocessing
3. YOLOv10 detection
4. Post-processing
5. Fuzzy logic severity calculation

**GPS Integration**
- NMEA protocol support
- Automatic port detection
- Real-time coordinate logging

**Cloud Storage**
- Efficient data transmission
- Metadata and image backup

## Troubleshooting

- **Camera not detected** - Check connections, verify permissions, try different camera index
- **Low detection accuracy** - Improve lighting, verify camera focus
- **GPS connection failed** - Check COM port, ensure clear sky view
- **Performance lag** - Enable GPU acceleration, reduce resolution, close background apps

## Related Projects

**[Road Defect Mapping Web App](https://github.com/rvnztolentino/road-defect-indexing-maps)**  
Interactive Mapbox dashboard for visualizing detected defects with GPS precision and road-type filtering. Part of the Comprehensive Road Defect Indexing System.

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- YOLO team for detection framework
- OpenCV and PyQt communities
- All contributors and maintainers

## Version

**v1.0.0** - Initial release with core detection, dashboard, GPS integration, and cloud storage support.

---

**Support**: Open an issue for questions or bug reports.