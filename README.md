# Road Defect Indexing System

A comprehensive system for detecting, analyzing, and mapping road defects using computer vision and GPS technology.

## Features

- Real-time road defect detection using YOLO object detection
- GPS-based defect location tracking
- Interactive map visualization of detected defects
- Defect severity calculation and analysis
- Configurable camera and detection settings
- Comprehensive logging system
- Modern GUI using CustomTkinter

## Project Structure

```
Road-Defect-Indexing-System/
├── src/
│   ├── .env                  # Environment variables (API keys, secrets)
│   └── app/
│       ├── gui/              # GUI components
│       ├── modules/          # Core functional modules
│       ├── utils/            # Utility functions
│       ├── config/           # Configuration files
│       ├── assets/           # Static assets
│       │   └── icons/        # Application icons
│       └── main.py           # Application entry point
├── models/                   # Trained model weights
├── logs/                     # Application logs
├── maps/                     # Generated maps
├── detections/              # Saved defect detections
├── requirements.txt         # Python dependencies
└── README.md               # Project documentation
```

## Environment Variables

Sensitive information such as API keys should be stored in the `.env` file located in the `src/` directory. Example:

```
CLOUD_API_KEY=your_cloud_api_key_here
MAPBOX_API_KEY=your_mapbox_api_key_here
OTHER_SERVICE_KEY=your_other_service_key_here
```

**Do not commit your `.env` file to version control.**

To use environment variables in your code, you can use the `os` and `dotenv` libraries:

```python
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path="src/.env")
api_key = os.getenv("CLOUD_API_KEY")
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Road-Defect-Indexing-System.git
cd Road-Defect-Indexing-System
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Copy `.env.example` to `.env` and fill in your API keys:
```bash
cp src/.env.example src/.env
```

## Usage

1. Start the application:
```bash
python src/app/main.py
```

2. Configure settings:
   - Camera settings (resolution, FPS)
   - GPS settings (baudrate, port)
   - Detection settings (confidence threshold)
   - Map settings (zoom level)

3. Start detection:
   - Click "Start Detection" to begin real-time defect detection
   - Detected defects will be marked on the map
   - Defect information and images will be saved

## Components

### GUI Components
- Main window with video feed
- Control panel for settings
- Map view for defect locations
- Status bar for system information

### Core Modules
- Object Detection: YOLO-based defect detection
- Camera: Video capture and processing
- GPS: Location tracking and data logging
- Severity Calculator: Defect analysis and scoring

### Utilities
- Configuration Management
- Logging System
- Map Generation
- Image Preprocessing

## Configuration

The system can be configured through the GUI or by editing the `config.json` file. Available settings include:

- Camera settings (resolution, FPS)
- Detection settings (model path, confidence threshold)
- GPS settings (baudrate, timeout)
- Map settings (save directory, zoom level)
- UI settings (theme, window size)

## Logging

The system maintains detailed logs in the `logs` directory:
- System events
- Camera status
- GPS data
- Detection results
- Error messages
- Performance metrics

## Maps

Generated maps are saved in the `maps` directory:
- Interactive HTML maps
- Defect markers with severity indicators
- Route tracking
- Exportable data in JSON format

## Requirements

- Python 3.8+
- OpenCV
- CustomTkinter
- Folium
- PyNMEA2
- Ultralytics (YOLO)
- python-dotenv
- Other dependencies listed in requirements.txt

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request 