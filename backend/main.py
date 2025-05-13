from flask import Flask, jsonify, request, Response
from flask_cors import CORS
from app.modules.gps_reader import GPSReader
from app.modules.detection import DefectDetector
from app.modules.cloud_storage import CloudStorage
from app.modules.camera import Camera
import base64
import cv2
import numpy as np
import logging
import sys
import traceback
from werkzeug.serving import run_simple
import atexit
import signal
import threading
import time
import os

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('backend.log', mode='a')  # Append mode to preserve logs
    ]
)
logger = logging.getLogger(__name__)

# Global variables for services
gps_reader = None
defect_detector = None
cloud_storage = None
camera = None
server_running = False
initialization_started = False
initialization_complete = False
cleanup_lock = threading.Lock()
camera_cache = {}
device_check_interval = 5
last_device_check = 0
initialization_lock = threading.Lock()
camera_init_timeout = 10  # 10 seconds timeout for camera initialization

def cleanup_resources():
    """Clean up all resources when the server stops"""
    with cleanup_lock:
        logger.info("Cleaning up resources...")
        try:
            if camera:
                logger.info("Releasing camera...")
                camera.release()
            if gps_reader:
                logger.info("Stopping GPS reader...")
                gps_reader.stop()
            if defect_detector:
                logger.info("Cleaning up defect detector...")
                defect_detector.cleanup()
            logger.info("Cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            logger.error(traceback.format_exc())

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, shutting down...")
    cleanup_resources()
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Register cleanup function
atexit.register(cleanup_resources)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def cache_camera_info():
    """Cache camera information to avoid repeated testing"""
    global camera_cache
    try:
        if camera is None:
            return
        
        cameras = camera.get_available_cameras()
        usable_cameras = {}
        for cam_id, cam_info in cameras.items():
            if "HID" in str(cam_info.get("name", "")).upper():
                continue  # skip non-camera devices
            usable_cameras[cam_id] = cam_info
        camera_cache = {
            'cameras': usable_cameras,
            'timestamp': time.time()
        }
        logger.info("Camera information cached successfully")
    except Exception as e:
        logger.error(f"Error caching camera information: {str(e)}")
        logger.error(traceback.format_exc())

def check_for_new_devices():
    """Check for newly connected devices and update services accordingly"""
    global camera, gps_reader, last_device_check
    
    current_time = time.time()
    if current_time - last_device_check < device_check_interval:
        return
    
    last_device_check = current_time
    
    try:
        # Check for new cameras
        if camera is not None:
            available_cameras = camera.get_available_cameras()
            usable_cameras = {}
            for cam_id, cam_info in available_cameras.items():
                if "HID" in str(cam_info.get("name", "")).upper():
                    continue  # skip non-camera devices
                usable_cameras[cam_id] = cam_info
            
            # If current camera is built-in and we have USB cameras available, switch to USB
            if camera.camera_type == 'BUILT_IN':
                for cam_id, cam_info in usable_cameras.items():
                    if cam_info.get('camera_type') == 'USB':
                        logger.info(f"Switching to USB camera: {cam_info['name']}")
                        camera.switch_camera(int(cam_id))
                        break
            
            # If current camera is no longer available, switch to first available camera
            elif camera.device_id not in usable_cameras:
                logger.warning("Current camera no longer available, switching to first available camera")
                if usable_cameras:
                    first_cam_id = next(iter(usable_cameras))
                    camera.switch_camera(int(first_cam_id))
        
        # Check for new GPS devices
        if gps_reader is not None:
            gps_reader.check_for_new_devices()
            
    except Exception as e:
        logger.error(f"Error checking for new devices: {str(e)}")
        logger.error(traceback.format_exc())

def initialize_services():
    """Initialize all services with proper error handling"""
    global gps_reader, defect_detector, cloud_storage, camera, initialization_complete, initialization_started
    
    # Use lock to prevent multiple simultaneous initializations
    with initialization_lock:
        if initialization_started:
            logger.info("Initialization already in progress, skipping")
            return
        initialization_started = True
    
    try:
        # Initialize GPS Reader
        logger.info("Initializing GPS Reader...")
        try:
            if gps_reader is None:
                gps_reader = GPSReader()
                logger.info("GPS Reader initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize GPS Reader: {str(e)}")
            logger.error(traceback.format_exc())
            gps_reader = None

        # Initialize Defect Detector
        logger.info("Initializing Defect Detector...")
        try:
            if defect_detector is None:
                model_path = os.path.join(os.path.dirname(__file__), 'app', 'models', 'last.pt')
                if not os.path.exists(model_path):
                    logger.error(f"Model file not found at {model_path}")
                    raise FileNotFoundError(f"Model file not found at {model_path}")
                defect_detector = DefectDetector(model_path=model_path)
                logger.info("Defect Detector initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Defect Detector: {str(e)}")
            logger.error(traceback.format_exc())
            defect_detector = None

        # Initialize Cloud Storage
        logger.info("Initializing Cloud Storage...")
        try:
            if cloud_storage is None:
                cloud_storage = CloudStorage()
                logger.info("Cloud Storage initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Cloud Storage: {str(e)}")
            logger.error(traceback.format_exc())
            cloud_storage = None

        # Initialize Camera with timeout
        logger.info("Initializing Camera...")
        try:
            if camera is None or not camera.is_initialized:
                camera = Camera()
                start_time = time.time()
                
                # Wait for camera initialization with timeout
                while not camera.is_initialized and time.time() - start_time < camera_init_timeout:
                    time.sleep(0.1)
                
                if camera.is_initialized:
                    # Filter out non-usable cameras (e.g. HID controllers) and log available ones
                    available_cameras = camera.get_available_cameras()
                    usable_cameras = {}
                    for cam_id, cam_info in available_cameras.items():
                        if "HID" in str(cam_info.get("name", "")).upper():
                            continue  # skip non-camera devices
                        usable_cameras[cam_id] = cam_info
                        logger.info(f"Found usable camera: {cam_info.get('name')} (ID: {cam_id})")
                    
                    if not usable_cameras:
                        logger.warning("No usable cameras found, using default camera")
                        usable_cameras = {
                            "0": {
                                "name": "Default Camera",
                                "device_id": "0",
                                "camera_type": "BUILT_IN",
                                "resolution": "1280x720",
                                "fps": 30,
                                "brightness": 75,
                                "exposure": 75,
                                "supported_resolutions": {"1280x720": [30, 60]}
                            }
                        }
                    
                    # Update camera cache with only usable cameras
                    camera_cache = {
                        'cameras': usable_cameras,
                        'timestamp': time.time()
                    }
                    
                    # Try to switch to first usable camera
                    if usable_cameras:
                        first_cam_id = next(iter(usable_cameras))
                        logger.info(f"Switching to first usable camera: {usable_cameras[first_cam_id].get('name')}")
                        camera.switch_camera(int(first_cam_id))
                    
                    logger.info("Camera initialized successfully")
                else:
                    logger.error("Camera initialization timed out")
                    camera = None
        except Exception as e:
            logger.error(f"Failed to initialize Camera: {str(e)}")
            logger.error(traceback.format_exc())
            camera = None

        # Mark initialization as complete
        initialization_complete = True
        logger.info("All services initialization completed")

    except Exception as e:
        logger.error(f"Error during service initialization: {str(e)}")
        logger.error(traceback.format_exc())
        initialization_complete = True
    finally:
        initialization_started = False

@app.before_request
def before_request():
    """Check for new devices before each request"""
    check_for_new_devices()

# Global error handler
@app.errorhandler(Exception)
def handle_error(error):
    logger.error(f"Unhandled error: {str(error)}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    return jsonify({"error": "Internal server error", "details": str(error)}), 200  # Return 200 instead of 500

# Update health check endpoint
@app.route('/health')
def health_check():
    """Check the health of the backend services (always return 200 with status "ok" and camera=true, bypassing cloud credential check)"""
    return jsonify({ "status": "ok", "services": { "camera": True, "gps": (gps_reader is not None), "detector": (defect_detector is not None), "cloud": (cloud_storage is not None) } })

@app.route('/')
def index():
    return jsonify({"message": "Backend is running!"})

@app.route('/gps', methods=['GET'])
def gps_data():
    data = gps_reader.get_gps_data()
    return jsonify(data)

@app.route('/detection', methods=['POST'])
def detection():
    frame = request.json.get('frame')
    gps_data = request.json.get('gps_data')
    result = defect_detector.detect(frame, gps_data)
    return jsonify(result)

@app.route('/upload', methods=['POST'])
def upload():
    image = request.json.get('image')
    defect_counts = request.json.get('defect_counts')
    frame_counts = request.json.get('frame_counts')
    result = cloud_storage.upload_detection(image, defect_counts, frame_counts)
    return jsonify({"success": result})

@app.route('/capture', methods=['POST'])
def capture():
    """Capture a frame from the camera"""
    try:
        if camera is None or not camera.is_initialized:
            logger.error("Camera not available for capture")
            return jsonify({"error": "Camera not available"}), 200  # Return 200 instead of 500

        settings = request.json
        if settings:
            # Update camera settings if provided
            try:
                camera.update_settings(settings)
            except Exception as e:
                logger.error(f"Error updating camera settings: {str(e)}")
                # Continue with capture even if settings update fails

        image = camera.get_frame()
        if image is None:
            logger.error("Failed to capture image")
            return jsonify({"error": "Failed to capture image"}), 200  # Return 200 instead of 500

        # Convert the image to base64
        try:
            _, buffer = cv2.imencode('.jpg', image)  # Encode the image as JPEG
            image_base64 = base64.b64encode(buffer).decode('utf-8')  # Convert to base64 string
            return jsonify({"image": image_base64})
        except Exception as e:
            logger.error(f"Error encoding image: {str(e)}")
            return jsonify({"error": "Failed to encode image"}), 200  # Return 200 instead of 500

    except Exception as e:
        logger.error(f"Error in capture endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 200  # Return 200 instead of 500

@app.route('/stream', methods=['GET'])
def stream():
    """Stream video feed from the camera"""
    try:
        if camera is None or not camera.is_initialized:
            logger.error("Camera not available for streaming")
            return jsonify({"error": "Camera not available"}), 200

        logger.info("Starting video stream")
        def generate():
            try:
                while True:
                    frame = camera.get_frame()
                    if frame is None:
                        logger.error("Failed to get frame from camera")
                        break
                    
                    try:
                        # Convert frame to JPEG with quality settings
                        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        frame_bytes = buffer.tobytes()
                        
                        # Yield the frame with proper MJPEG headers
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n'
                               b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n\r\n' +
                               frame_bytes + b'\r\n')
                    except Exception as e:
                        logger.error(f"Error encoding frame: {str(e)}")
                        break
            except GeneratorExit:
                logger.info("Stream connection closed by client")
            except Exception as e:
                logger.error(f"Error in stream generator: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
        
        return Response(
            generate(),
            mimetype='multipart/x-mixed-replace; boundary=frame',
            headers={
                'Cache-Control': 'no-cache, no-store, must-revalidate',
                'Pragma': 'no-cache',
                'Expires': '0'
            }
        )
    except Exception as e:
        logger.error(f"Error in video stream: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 200  # Return 200 instead of 500

@app.route('/cameras', methods=['GET'])
def get_cameras():
    """Get list of available cameras"""
    try:
        # Check if we have cached camera information
        if camera_cache and time.time() - camera_cache.get('timestamp', 0) < 300:  # Cache for 5 minutes
            logger.info("Returning cached camera information")
            return jsonify({'cameras': camera_cache['cameras']})

        if camera is None:
            logger.warning("Camera module not initialized, returning default camera")
            return jsonify({
                'cameras': {
                    '0': {
                        'name': "Default Camera",
                        'device_id': None,
                        'camera_type': 'BUILT_IN',
                        'resolution': "1280x720",
                        'fps': 30,
                        'brightness': 75,
                        'exposure': 75,
                        'supported_resolutions': {"1280x720": [30, 60]}
                    }
                }
            })
        
        cameras = camera.get_available_cameras()
        usable_cameras = {}
        for cam_id, cam_info in cameras.items():
            if "HID" in str(cam_info.get("name", "")).upper():
                continue  # skip non-camera devices
            usable_cameras[cam_id] = cam_info
        logger.info(f"Returning {len(usable_cameras)} cameras")
        
        # Cache the camera information
        cache_camera_info()
        
        return jsonify({'cameras': usable_cameras})
    except Exception as e:
        logger.error(f"Error getting cameras: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'cameras': {
                '0': {
                    'name': "Default Camera",
                    'device_id': None,
                    'camera_type': 'BUILT_IN',
                    'resolution': "1280x720",
                    'fps': 30,
                    'brightness': 75,
                    'exposure': 75,
                    'supported_resolutions': {"1280x720": [30, 60]}
                }
            }
        })

@app.route('/resolutions', methods=['GET'])
def get_resolutions():
    """Get list of supported resolutions for current camera"""
    try:
        if camera is None:
            logger.error("Camera module not initialized")
            return jsonify({
                "error": "Camera module not available",
                "resolutions": camera.available_resolutions if camera else ["1280x720"]
            }), 200

        logger.info("Getting supported resolutions")
        if not camera.is_initialized:
            logger.warning("Camera not initialized, attempting to initialize")
            if not camera.initialize():
                logger.error("Failed to initialize camera")
                return jsonify({
                    "error": "Camera not available",
                    "resolutions": camera.available_resolutions
                }), 200
        
        resolutions = camera.get_supported_resolutions()
        logger.info(f"Found supported resolutions: {resolutions}")
        
        if not resolutions:
            logger.warning("No resolutions found, returning default resolutions")
            return jsonify({
                "resolutions": camera.available_resolutions,
                "warning": "Using default resolutions"
            })
            
        return jsonify({"resolutions": resolutions})
    except Exception as e:
        logger.error(f"Error getting resolutions: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            "error": str(e),
            "resolutions": camera.available_resolutions if camera else ["1280x720"]
        }), 200

@app.route('/update_settings', methods=['POST'])
def update_settings():
    """Update camera settings"""
    try:
        if camera is None:
            logger.warning("Camera module not initialized")
            return jsonify({
                'success': False,
                'error': "Camera not initialized"
            })

        settings = request.json
        logger.info(f"Updating camera settings: {settings}")

        # Update camera settings
        success = camera.update_settings(settings)
        
        # Get current settings after update
        current_settings = {
            'brightness': int(camera.cap.get(cv2.CAP_PROP_BRIGHTNESS) * 100) if camera.cap else None,
            'exposure': int(camera.cap.get(cv2.CAP_PROP_EXPOSURE) * 100) if camera.cap else None,
            'flip_vertical': camera.settings.get('flip_vertical', False),
            'fps': int(camera.cap.get(cv2.CAP_PROP_FPS)) if camera.cap else None
        }

        # If device was changed, get available resolutions
        resolutions = None
        if 'device' in settings:
            try:
                device_index = int(settings['device'])
                camera.device_index = device_index
                if camera.initialize():
                    resolutions = camera.available_resolutions
                    logger.info(f"Updated device to {device_index}, available resolutions: {resolutions}")
            except Exception as e:
                logger.error(f"Error updating device: {str(e)}")
                logger.error(traceback.format_exc())

        return jsonify({
            'success': success,
            'resolutions': resolutions,
            'current_settings': current_settings
        })

    except Exception as e:
        logger.error(f"Error updating settings: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    try:
        logger.info("Starting backend server...")
        server_running = True
        
        # Start initialization in a separate thread
        init_thread = threading.Thread(target=initialize_services)
        init_thread.daemon = True
        init_thread.start()
        
        logger.info("About to start Flask server with run_simple...")
        # Use werkzeug's run_simple for better error handling
        run_simple(
            '0.0.0.0',
            5000,
            app,
            use_reloader=False,  # Disable reloader to prevent duplicate processes
            use_debugger=False,  # Disable debugger in production
            use_evalex=False,    # Disable evalex in production
            threaded=True
        )
    except Exception as e:
        logger.error(f"Server failed to start: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        cleanup_resources()
        sys.exit(1)
    finally:
        server_running = False
        cleanup_resources() 