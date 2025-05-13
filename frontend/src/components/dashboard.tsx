"use client"

import { useState, useEffect, useRef, SetStateAction } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Slider } from "@/components/ui/slider"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Input } from "@/components/ui/input"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Checkbox } from "@/components/ui/checkbox"
import { Label } from "@/components/ui/label"
import {
  Play,
  Settings,
  MapPin,
  Folder,
  Calendar,
  Trash2,
  Sliders,
  Filter,
  Compass,
  Camera,
  FlipVertical,
  Cloud,
  AlertCircle,
  CheckCircle,
  Info,
  BarChart2,
} from "lucide-react"
import { useRouter } from "next/navigation"
import { detectionApi } from "@/lib/api"
import axios from 'axios'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog"
import { Progress } from "@/components/ui/progress"

type LogType = "info" | "success" | "error" | "warning"

interface LogMessage {
  id: number
  type: LogType
  message: string
  timestamp: string
}

// Add interfaces for API responses
interface CamerasResponse {
  cameras: {
    [key: string]: string | {
      camera_type: any
      name: string;
      device_id?: string;
      resolution?: string;
      fps?: number;
      brightness?: number;
      exposure?: number;
      supported_resolutions?: {
        [key: string]: number[];
      };
    };
  };
}

interface ResolutionsResponse {
  resolutions: string[];
}

interface UpdateSettingsResponse {
  success: boolean;
  error?: string;
  resolutions?: string[];
  current_settings?: {
    brightness?: number;
    exposure?: number;
    flip_vertical?: boolean;
    fps?: number;
  };
}

// Add types for API responses
interface CaptureResponse {
  image: string;
}

interface DetectionResponse {
  defects?: {
    linear: number;
    alligator: number;
    pothole: number;
  };
  stats?: {
    cpu: number;
    gpu: number;
    detection_time: number;
    fps: number;
  };
  image?: string;
}

interface GpsResponse {
  longitude: string;
  latitude: string;
}

// Add new interfaces for camera and resolution data
interface CameraDevice {
  device_id: string;
  name: string;
  camera_type: string;
  resolution: string;
  fps: number;
  brightness: number;
  exposure: number;
  supported_resolutions: { [key: string]: number[] };
}

// Update the VideoSettings interface
interface VideoSettings {
  device: string;
  resolution: string;
  brightness: number;
  exposure: number;
  flip_vertical: boolean;
  zoom: number;
  fps: number;
}

interface CameraSettings {
  device: string;
  resolution: string;
  fps: number;
  brightness: number;
  exposure: number;
  flip_vertical: boolean;
  zoom: number;
}

export default function Dashboard() {
  const router = useRouter()
  // Video stream state
  const [isPlaying, setIsPlaying] = useState(false)
  const imgRef = useRef<HTMLImageElement>(null)
  const [liveFrame, setLiveFrame] = useState<string>("")
  const [brightness, setBrightness] = useState(50)
  const [exposure, setExposure] = useState(50)
  const [cameraDevice, setCameraDevice] = useState("default")
  const [resolution, setResolution] = useState("720p")
  const [fps, setFps] = useState("30")
  const [flipVertical, setFlipVertical] = useState(false)
  const [zoom, setZoom] = useState(1.0)
  const [selectedFps, setSelectedFps] = useState<number>(30)

  // Detection state
  const [detectionActive, setDetectionActive] = useState(false)
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.5)
  const [classFilter, setClassFilter] = useState(["linear", "alligator", "pothole"])
  const [gpsConnected, setGpsConnected] = useState(false)
  const [gpsConnecting, setGpsConnecting] = useState(false)

  // Upload state
  const [bucketPath, setBucketPath] = useState("s3://my-bucket/detections/")
  const [cloudConnected, setCloudConnected] = useState(false)
  const [cloudConnecting, setCloudConnecting] = useState(false)

  // Settings state
  const [defaultOutputPath, setDefaultOutputPath] = useState("/output")
  const [organizeByDate, setOrganizeByDate] = useState(true)
  const [autoDeleteRaw, setAutoDeleteRaw] = useState(false)

  // Statistics state
  const [linearCracks, setLinearCracks] = useState(0)
  const [alligatorCracks, setAlligatorCracks] = useState(0)
  const [potholes, setPotholes] = useState(0)
  const [gpsData, setGpsData] = useState({ longitude: "N/A", latitude: "N/A" })

  // System stats
  const [cpuUsage, setCpuUsage] = useState(0)
  const [gpuUsage, setGpuUsage] = useState(0)
  const [detectionTime, setDetectionTime] = useState(0)
  const [currentFps, setCurrentFps] = useState(0)

  // Settings modal
  const [showSettings, setShowSettings] = useState(false)

  // Console logs
  const [logs, setLogs] = useState<LogMessage[]>([])
  const logIdCounter = useRef(0)
  const logsEndRef = useRef<HTMLDivElement>(null)

  // Fix hydration error: only show time after mount
  const [mounted, setMounted] = useState(false)
  useEffect(() => { setMounted(true) }, [])

  const addLog = (type: LogType, message: string) => {
    const now = new Date()
    const timestamp = `${String(now.getHours()).padStart(2, "0")}:${String(now.getMinutes()).padStart(2, "0")}:${String(now.getSeconds()).padStart(2, "0")}`

    const newLog: LogMessage = {
      id: logIdCounter.current++,
      type,
      message,
      timestamp,
    }

    setLogs((prev) => [...prev.slice(-9), newLog])
  }

  // Scroll logs to bottom when new logs are added
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [logs])

  const togglePlay = () => {
    const newIsPlaying = !isPlaying;
    setIsPlaying(newIsPlaying);
    if (newIsPlaying) {
      addLog("info", "Video stream started");
    } else {
      stopStream();
    }
  }

  // Update video stream state and refs
  const videoRef = useRef<HTMLVideoElement>(null)

  // Update useEffect for video stream
  useEffect(() => {
    if (imgRef.current) {
      if (isPlaying) {
        // Set the image source to the stream URL with a timestamp to prevent caching
        const streamUrl = `http://localhost:5000/stream?t=${Date.now()}`
        imgRef.current.src = streamUrl
        addLog("info", "Video stream started")
      } else {
        // Stop the stream by removing the source
        if (imgRef.current.src) {
          imgRef.current.src = ''
        }
        addLog("info", "Video stream stopped")
      }
    }

    return () => {
      // Cleanup when component unmounts
      if (imgRef.current) {
        imgRef.current.src = ''
      }
    }
  }, [isPlaying])

  const toggleDetection = async () => {
    if (!detectionActive) {
      addLog("info", "Starting detection...");
      setDetectionActive(true);

      try {
        // Start detection by sending a request to the backend
        const detectionResponse = await axios.post<DetectionResponse>('http://localhost:5000/detection', {
          frame: liveFrame, // Adjust based on how you capture frames
          gps_data: gpsData
        });

        // Handle the detection response
        const { defects, stats } = detectionResponse.data;
        if (defects) {
          setLinearCracks(defects.linear);
          setAlligatorCracks(defects.alligator);
          setPotholes(defects.pothole);
        }
        if (stats) {
          setCpuUsage(stats.cpu);
          setGpuUsage(stats.gpu);
          setDetectionTime(stats.detection_time);
          setCurrentFps(stats.fps);
        }
      } catch (error) {
        addLog("error", "Failed to start detection");
        setDetectionActive(false);
      }
    } else {
      setDetectionActive(false);
      addLog("info", "Detection stopped");
    }
  };

  const toggleFlipVertical = async () => {
    const newFlipState = !flipVertical
    setFlipVertical(newFlipState)
    await handleVideoControls({ flip_vertical: newFlipState })
    addLog("info", newFlipState ? "Video flip enabled" : "Video flip disabled")
  }

  const toggleGps = async () => {
    if (!gpsConnected && !gpsConnecting) {
      setGpsConnecting(true);
      addLog("info", "Connecting to GPS...");

      try {
        // Fetch GPS data from the backend
        const response = await axios.get<GpsResponse>('http://localhost:5000/gps'); // Specify the response type
        const data = response.data;

        if (data && typeof data.longitude === 'string' && typeof data.latitude === 'string') {
          setGpsData(data); // This should now work without linter errors
          setGpsConnected(true);
          addLog("success", "GPS connected successfully");
        } else {
          // Set default GPS data if invalid
          setGpsData({ longitude: "0", latitude: "0" });
          addLog("warning", "Invalid GPS data received, using default values.");
            }
          } catch (error) {
        // Set default GPS data on error
        setGpsData({ longitude: "0", latitude: "0" });
        addLog("error", "Failed to connect to GPS, using default values.");
      } finally {
        setGpsConnecting(false);
      }
    } else if (gpsConnected) {
      setGpsConnected(false);
      addLog("info", "GPS disconnected");
    }
  };

  const testCloudConnection = () => {
    setCloudConnecting(true)
    addLog("info", "Testing cloud connection...")

    // Simulate connection test
    setTimeout(() => {
      const success = Math.random() > 0.3 // 70% success rate for demo
      setCloudConnecting(false)

      if (success) {
        setCloudConnected(true)
        addLog("success", "Cloud connection successful")
      } else {
        setCloudConnected(false)
        addLog("error", "Cloud connection failed. Check credentials.")
      }
    }, 1500)
  }

  const navigateToAnalysis = () => {
    router.push("/analysis")
  }

  // Format time as HH:MM:SS
  const formatTime = () => {
    const now = new Date()
    return `${String(now.getHours()).padStart(2, "0")}:${String(now.getMinutes()).padStart(2, "0")}:${String(now.getSeconds()).padStart(2, "0")}`
  }

  useEffect(() => {
    let interval: NodeJS.Timeout | null = null;
    if (isPlaying) {
      const pollCamera = async () => {
         try {
          const cameraSettings = {
            device: cameraDevice,
            resolution,
            fps: parseInt(fps),
            brightness,
            exposure,
            flip_vertical: flipVertical,
            zoom
          };
           const captureResult = await detectionApi.capture(cameraSettings) as CaptureResponse;
           if (captureResult && captureResult.image) {
             setLiveFrame(captureResult.image);
           }
         } catch (err) {
           console.error("Error polling camera:", err);
         }
      };
      // Poll every 100ms (adjust as needed) if isPlaying is true.
      interval = setInterval(pollCamera, 100);
    } else {
       // Clear interval (and reset liveFrame) if isPlaying is false.
       setLiveFrame("");
    }
    return () => { if (interval) clearInterval(interval); };
  }, [isPlaying, cameraDevice, resolution, fps, brightness, exposure, flipVertical, zoom]);

  // Settings state
  const [settings, setSettings] = useState<CameraSettings>({
    device: '',
    resolution: '',
    fps: 30,
    brightness: 50,
    exposure: 50,
    flip_vertical: false,
    zoom: 1
  });

  // Add new state for cameras and resolutions
  const [availableCameras, setAvailableCameras] = useState<CameraDevice[]>([])
  const [availableResolutions, setAvailableResolutions] = useState<string[]>([])
  const [selectedCamera, setSelectedCamera] = useState<string>("")
  const [selectedResolution, setSelectedResolution] = useState<string>("")

  // Add new state variables after other state declarations
  const [isBackendInitializing, setIsBackendInitializing] = useState(true)
  const [initializationProgress, setInitializationProgress] = useState(0)
  const [initializationStatus, setInitializationStatus] = useState("Initializing backend services...")
  const [cameraLoading, setCameraLoading] = useState(false)

  // Add state for retry count and health check
  const [backendRetryCount, setBackendRetryCount] = useState(0);
  const [healthCheckTimeout, setHealthCheckTimeout] = useState<NodeJS.Timeout | null>(null);

  // Add new function to check backend health
  const checkBackendHealth = async () => {
    try {
      // Poll health endpoint with a short timeout (500ms) so that we do not hang waiting for "ok" or camera service.
      const response = await axios.get('http://127.0.0.1:5000/health', { timeout: 500, headers: { 'Cache-Control': 'no-cache', 'Pragma': 'no-cache' } });
      // If we get a 200 (even if status is "initializing"), immediately hide the loading dialog and fetch cameras.
      setInitializationProgress(100);
      setInitializationStatus("Backend (health endpoint) responded; proceeding...");
      setTimeout(() => { setIsBackendInitializing(false); fetchCameras(); }, 500);
    } catch (error) {
      console.error("Backend health check failed (or timed out):", error);
      setInitializationStatus("Error (or timeout) connecting to backend. Retrying...");
      setInitializationProgress(0);
      const retryDelay = Math.min(2000 * Math.pow(1.5, backendRetryCount), 10000);
      setBackendRetryCount(prev => prev + 1);
      setTimeout(checkBackendHealth, retryDelay);
    }
  };

  // Update useEffect for backend health check
  useEffect(() => {
    let isMounted = true;

    const startHealthCheck = () => {
      if (isMounted) {
        checkBackendHealth();
      }
    };

    // Start health check immediately
    startHealthCheck();

    // Cleanup function
    return () => {
      isMounted = false;
      if (healthCheckTimeout) {
        clearTimeout(healthCheckTimeout);
      }
    };
  }, []); // Empty dependency array since we want this to run once on mount

  // Update LoadingDialog component
  const LoadingDialog = () => (
    <Dialog open={isBackendInitializing} modal={true}>
      <DialogContent className="sm:max-w-md bg-[#1a1a1a] border-[#333]">
        <DialogHeader>
          <DialogTitle className="text-white text-xl">Initializing Backend</DialogTitle>
          <DialogDescription className="text-gray-300 mt-2">
            {initializationStatus}
          </DialogDescription>
        </DialogHeader>
        <div className="py-4">
          <Progress value={initializationProgress} className="w-full h-2 bg-[#333]" />
          <div className="mt-4 text-center">
            {initializationProgress === 0 && (
              <div className="space-y-2">
                <p className="text-sm text-gray-400">
                  Waiting for backend server at http://localhost:5000
                </p>
                <p className="text-xs text-gray-500">
                  {backendRetryCount > 0 ? `Retry attempt ${backendRetryCount}...` : 'This may take a few moments...'}
                </p>
              </div>
            )}
            {initializationProgress > 0 && initializationProgress < 100 && (
              <p className="text-sm text-gray-400">
                Initializing services... {Math.round(initializationProgress)}%
              </p>
            )}
            {initializationProgress === 100 && (
              <p className="text-sm text-green-400">
                Backend initialized successfully!
              </p>
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );

  // Update the fetchCameras function
  const fetchCameras = async () => {
    try {
      setCameraLoading(true);
      const response = await axios.get<{ cameras: CameraDevice[] }>('http://127.0.0.1:5000/cameras');
      const cameras = response.data.cameras; // Correctly access the cameras array
      console.log('Fetched cameras:', cameras); // Log the fetched camera data
      
      if (cameras && Object.keys(cameras).length > 0) { // Check if cameras object is not empty
        setAvailableCameras(Object.values(cameras)); // Convert object to array
        // Select first camera by default if none selected
        if (!selectedCamera) {
          const firstCamera = Object.values(cameras)[0];
          setSelectedCamera(firstCamera.device_id);
          // Set initial camera settings
          setBrightness(firstCamera.brightness || 50);
          setExposure(firstCamera.exposure || 50);
          setZoom(1.0); // Ensure zoom is set to default
          // Set initial resolution
          const resolutions = Object.keys(firstCamera.supported_resolutions || {});
          if (resolutions.length > 0) {
            setSelectedResolution(resolutions[0]);
            setFps(firstCamera.supported_resolutions[resolutions[0]][0].toString());
          }
        }
      } else {
        setAvailableCameras([]);
        addLog("warning", "No cameras found, retrying...");
        // Retry fetching cameras after a delay
        setTimeout(fetchCameras, 2000);
      }
    } catch (err: unknown) {
      const error = err as Error;
      console.error("Error fetching cameras:", error);
      addLog("error", `Failed to fetch cameras: ${error.message}`);
      setAvailableCameras([]);
    } finally {
      setCameraLoading(false);
    }
  };

  // Add zoom handler
  const handleZoomChange = async (value: number[]) => {
    const newZoom = value[0];
    setZoom(newZoom);
    await handleVideoControls({ zoom: newZoom });
  };

  // Update handleVideoControls function
  const handleVideoControls = async (newSettings: Partial<CameraSettings>) => {
    try {
      setCameraLoading(true);
      const updatedSettings = { ...settings, ...newSettings };
      setSettings(updatedSettings);
      
      const response = await axios.post('http://localhost:5000/camera/controls', updatedSettings);
      if (response.status === 200) {
        const camera = availableCameras.find(c => c.device_id === updatedSettings.device);
        if (camera) {
          setBrightness(updatedSettings.brightness || camera.brightness);
          setExposure(updatedSettings.exposure || camera.exposure);
          setZoom(updatedSettings.zoom || 1.0);
        }
      }
    } catch (error) {
      console.error("Error updating video controls:", error);
      addLog("error", "Failed to update camera settings");
    } finally {
      setCameraLoading(false);
    }
  };

  // Update the brightness handler
  const handleBrightnessChange = async (value: number[]) => {
    const newBrightness = value[0];
    setBrightness(newBrightness);
    await handleVideoControls({ brightness: newBrightness });
  };

  // Update the exposure handler
  const handleExposureChange = async (value: number[]) => {
    const newExposure = value[0];
    setExposure(newExposure);
    await handleVideoControls({ exposure: newExposure });
  };

  // Update the resolution selection component
  const ResolutionSelect = () => {
    const currentCamera = availableCameras.find(c => c.device_id === selectedCamera);
    const resolutions = currentCamera?.supported_resolutions || {};

    const handleResolutionChange = async (newResolution: string) => {
      try {
        setResolution(newResolution);
        // Reset fps to first available option for new resolution
        const newFps = resolutions[newResolution][0].toString();
        setFps(newFps);
        setSelectedFps(parseInt(newFps));
        await handleVideoControls({
          device: selectedCamera,
          resolution: newResolution,
          fps: parseInt(newFps),
          brightness,
          exposure,
          flip_vertical: flipVertical,
          zoom
        });
      } catch (err) {
        console.error("Error changing resolution:", err);
      }
    };

    return (
      <div className="space-y-2 w-full">
        <Label htmlFor="resolution" className="text-sm font-medium text-white">Resolution</Label>
        <Select 
          value={`${selectedResolution}@${selectedFps}fps`}
          onValueChange={async (value: string) => {
            try {
              setCameraLoading(true);
              const [res, fps] = value.split('@');
              setSelectedResolution(res);
              setSelectedFps(parseInt(fps));
              await handleResolutionChange(res);
            } catch (error) {
              console.error("Error handling resolution change:", error);
            } finally {
              setCameraLoading(false);
            }
          }}
          disabled={cameraLoading || !currentCamera}
        >
          <SelectTrigger id="resolution" className="w-full h-9 bg-[#1a1a1a] border-[#333] text-white">
            <SelectValue placeholder="Select resolution" />
          </SelectTrigger>
          <SelectContent className="bg-[#1a1a1a] border-[#333] text-white">
            {Object.keys(resolutions).length === 0 ? (
              <SelectItem value="default" disabled>No resolutions available</SelectItem>
            ) : (
              Object.entries(resolutions).map(([res, fpsList]) => 
                fpsList.map(fps => (
                  <SelectItem key={`${res}@${fps}`} value={`${res}@${fps}`} className="text-white">
                    {res} @ {fps}fps
                  </SelectItem>
                ))
              )
            )}
          </SelectContent>
        </Select>
      </div>
    );
  };

  const handleCameraChange = async (newCamera: string) => {
    setSelectedCamera(newCamera);
    const selectedCam = availableCameras.find(cam => cam.device_id === newCamera);
    if (selectedCam) {
      // Convert the keys of supported_resolutions to an array of strings
      setAvailableResolutions(Object.keys(selectedCam.supported_resolutions));
      setSelectedResolution(selectedCam.resolution);
      await updateCameraSettings(selectedCam);
    }
  };

  const updateCameraSettings = async (camera: CameraDevice) => {
    setCameraLoading(true);
    try {
      await handleVideoControls({
        device: camera.device_id,
        resolution: selectedResolution,
        fps: parseInt(fps),
        brightness,
        exposure,
        flip_vertical: flipVertical,
        zoom
      });
    } catch (error) {
      console.error("Error updating camera settings:", error);
    } finally {
      setCameraLoading(false);
    }
  };

  // Update camera selection component
  const CameraSelect = () => (
    <div className="space-y-2 w-full">
      <Label htmlFor="camera" className="text-sm font-medium text-white">Camera</Label>
      <Select
        value={selectedCamera}
        onValueChange={handleCameraChange}
        disabled={cameraLoading}
      >
        <SelectTrigger id="camera" className="w-full h-9 bg-[#1a1a1a] border-[#333] text-white">
          <SelectValue placeholder="Select camera" />
        </SelectTrigger>
        <SelectContent className="bg-[#1a1a1a] border-[#333] text-white">
          {availableCameras.length === 0 ? (
            <SelectItem value="default" disabled>No cameras found</SelectItem>
          ) : (
            availableCameras.map((camera) => (
              <SelectItem key={camera.device_id} value={camera.device_id} className="text-white">
                {camera.name} ({camera.camera_type})
              </SelectItem>
            ))
          )}
        </SelectContent>
      </Select>
    </div>
  );

  // Add a function to stop the camera stream
  const stopStream = () => {
    setIsPlaying(false);
    if (imgRef.current) {
      imgRef.current.src = '';
    }
    addLog("info", "Video stream stopped");
  };

  return (
    <div className="h-screen flex flex-col bg-[#0a0a0a] overflow-hidden">
      <LoadingDialog />
      {/* Add a semi-transparent overlay when backend is initializing */}
      {isBackendInitializing && (
        <div className="fixed inset-0 bg-black/50 z-40" />
      )}
      
      {/* Top section with video and map */}
      <div className="flex flex-col lg:flex-row h-auto flex-grow overflow-hidden">
        {/* Video Stream Panel */}
        <div className="w-full lg:w-[60%] h-[40vh] lg:h-full bg-[#1a1a1a] relative">
          <div className="h-10 bg-[#0a0a0a] flex items-center p-4">
            <Camera className="h-6 w-6 text-white mr-2" />
            <span className="text-base text-white">Video Stream</span>
            <div className="ml-auto bg-[#050e39] text-white text-sm px-3 py-1 rounded-sm">
              {detectionActive ? "DETECTING" : "STANDBY"}
            </div>
          </div>
          <div className="h-[calc(100%-40px)] bg-[#1a1a1a] flex items-center justify-center relative">
            {!isPlaying ? (
              <div className="text-center">
                <Camera className="w-16 h-16 text-white/30 mx-auto mb-2" />
                <p className="text-white/50 text-sm">Press Start Live Feed to begin</p>
            <Button
                  variant="outline"
                  className="mt-4 bg-[#050e39] hover:bg-[#0a1a5a] text-white"
                  onClick={() => setIsPlaying(true)}
                  disabled={!selectedCamera || !selectedResolution}
                >
                  <Play className="h-4 w-4 mr-2" />
                  Start Live Feed
            </Button>
              </div>
            ) : (
              <img
                ref={imgRef}
                className="max-w-full max-h-full object-contain"
                style={{
                  transform: flipVertical ? 'scaleY(-1)' : 'none'
                }}
                alt="Live video stream"
              />
            )}
          </div>
        </div>

        {/* Mapbox View Panel */}
        <div className="w-full lg:w-[40%] h-[30vh] lg:h-[65vh] bg-[#1a1a1a] relative mt-2 lg:mt-0 lg:ml-2">
          <div className="h-10 bg-[#0a0a0a] flex items-center p-4">
            <MapPin className="h-6 w-6 text-white mr-2" />
            <span className="text-base text-white">Map View</span>
            <div className="ml-auto text-white text-sm">GPS: {gpsConnected ? "CONNECTED" : "DISCONNECTED"}</div>
          </div>
          <div className="h-[calc(100%-40px)] bg-[#e5e7eb] flex items-center justify-center">
            {gpsConnected ? (
              <div className="text-center">
                <MapPin className="w-8 h-8 text-[#050e39] mx-auto" />
                <p className="text-[#333] text-xs mt-2">Map would be displayed here</p>
              </div>
            ) : (
              <div className="text-center">
                <MapPin className="w-8 h-8 text-gray-400 mx-auto" />
                <p className="text-gray-500 text-xs mt-2">GPS not connected</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Controls Section */}
      <div className={`grid grid-cols-1 lg:grid-cols-3 gap-2 lg:gap-0 mt-2 flex-shrink-0 ${isBackendInitializing ? 'pointer-events-none opacity-50' : ''}`}>
        {/* Video Control Panel */}
        <div className="border-r border-[#222] flex flex-col bg-[#0a0a0a]">
          <div className="h-8 bg-[#0a0a0a] flex items-center px-4 border-b border-[#222]">
            <span className="text-base text-white">Video Controls</span>
          </div>
          <div className="p-4 flex-1 flex flex-col justify-between">
            <div className="grid grid-row gap-4 ml-1 p-2">
              <CameraSelect />
            </div>

            <div className="space-y-4 mt-4 ml-3 mr-3">
            <div className="mr-4">
              <div className="flex flex-col sm:flex-row sm:justify-between items-center gap-4 w-full mr-4">
                  <Button
                    variant="outline"
                    size="sm"
                    className={`h-10 px-4 w-full sm:w-1/2 ${flipVertical ? 'bg-[#050e39] text-white' : 'bg-[#1a1a1a] text-white'}`}
                    onClick={toggleFlipVertical}
                  >
                    <FlipVertical className="h-4 w-4 mr-2" />
                    Flip Vertical
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    className="h-10 px-4 w-full sm:w-1/2 bg-[#1a1a1a] text-white"
                    onClick={stopStream}
                  >
                    Stop Stream
                  </Button>
                </div>
            </div>
            

              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-white">Zoom</span>
                  <span className="text-sm text-white">{Math.round(zoom * 100)}%</span>
                </div>
                <Slider
                  value={[zoom]}
                  onValueChange={handleZoomChange}
                  min={0.1}
                  max={5.0}
                  step={0.1}
                  className="py-1"
                />
              </div>

              <div className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="text-sm text-white">Brightness</span>
                <span className="text-sm text-white">{brightness}%</span>
              </div>
              <Slider
                value={[brightness]}
                  onValueChange={handleBrightnessChange}
                max={100}
                step={1}
                className="py-1"
              />
            </div>

              <div className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="text-sm text-white">Exposure</span>
                <span className="text-sm text-white">{exposure}%</span>
              </div>
              <Slider
                value={[exposure]}
                  onValueChange={handleExposureChange}
                max={100}
                step={1}
                className="py-1"
              />
              </div>
            </div>
          </div>
        </div>

        {/* Statistics Panel */}
        <div className="border-r border-[#222] flex flex-col bg-[#0a0a0a]">
          <div className="h-8 bg-[#0a0a0a] flex items-center px-4 border-b border-[#222]">
            <span className="text-base text-white">Statistics Logs</span>
          </div>
          <div className="p-4 flex-1 flex flex-col justify-between">
            <div className="grid grid-cols-3 gap-2">
              <div className="flex flex-col items-center">
                <div className="w-12 h-12 rounded-full border-2 border-[#333] flex items-center justify-center mb-1 bg-[#1a1a1a]">
                  <span className="text-white text-lg font-bold">{linearCracks}</span>
                </div>
                <p className="text-base text-white text-center">Linear Cracks</p>
              </div>
              <div className="flex flex-col items-center">
                <div className="w-12 h-12 rounded-full border-2 border-[#333] flex items-center justify-center mb-1 bg-[#1a1a1a]">
                  <span className="text-white text-lg font-bold">{alligatorCracks}</span>
                </div>
                <p className="text-base text-white text-center">Alligator Cracks</p>
              </div>
              <div className="flex flex-col items-center">
                <div className="w-12 h-12 rounded-full border-2 border-[#333] flex items-center justify-center mb-1 bg-[#1a1a1a]">
                  <span className="text-white text-lg font-bold">{potholes}</span>
                </div>
                <p className="text-base text-white text-center">Potholes</p>
              </div>
            </div>

            {/* Improved GPS Data Display with dot indicator */}
            <div className="bg-[#1a1a1a] p-3 rounded-sm mt-4">
              <p className="text-base text-white mb-2 flex items-center">
                <Compass className="h-4 w-4 mr-2" />
                GPS Data{" "}
                <span className="ml-auto flex items-center">
                  <span
                    className={`inline-block w-3 h-3 rounded-full mr-1 ${
                      gpsConnected ? "bg-green-500" : gpsConnecting ? "bg-yellow-500" : "bg-red-500"
                    }`}
                  ></span>
                </span>
              </p>

              <div className="grid grid-cols-2 gap-2">
                <div className="bg-[#0a0a0a] rounded p-2 text-center">
                  <div className="text-xs text-gray-400 mb-1">LONG</div>
                  <div className="text-sm text-white font-mono">{gpsData.longitude}</div>
                </div>
                <div className="bg-[#0a0a0a] rounded p-2 text-center">
                  <div className="text-xs text-gray-400 mb-1">LAT</div>
                  <div className="text-sm text-white font-mono">{gpsData.latitude}</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Main Control Panel */}
        <div className="flex flex-col bg-[#0a0a0a]">
          <div className="h-8 bg-[#0a0a0a] flex items-center px-4 border-b border-[#222]">
            <span className="text-base text-white">Main Controls</span>
          </div>
          <div className="p-4 flex-1 flex flex-col justify-between">
            <Button
              variant="outline"
              className={`w-full h-10 ${
                detectionActive ? "bg-red-600 hover:bg-red-700" : "bg-[#050e39] hover:bg-[#0a1a5a]"
              } text-white border-0 flex items-center`}
              onClick={toggleDetection}
            >
              <Play className="h-4 w-4 mr-2 flex-shrink-0" />
              <span className="text-base truncate mx-auto">
                {detectionActive ? "Stop Detection" : "Start Live Detection"}
              </span>
            </Button>

            <Button
              variant="outline"
              className="w-full h-10 bg-[#1a1a1a] hover:bg-[#252525] text-white border-[#333] flex items-center mt-4"
              onClick={toggleGps}
            >
              <Compass className="h-4 w-4 mr-2 flex-shrink-0" />
              <span className="text-base truncate mx-auto">{gpsConnected ? "GPS Connected" : "Connect GPS"}</span>
            </Button>

            <Button
              variant="outline"
              className="w-full h-10 bg-[#1a1a1a] hover:bg-[#252525] text-white border-[#333] flex items-center mt-4"
              onClick={navigateToAnalysis}
            >
              <BarChart2 className="h-4 w-4 mr-2 flex-shrink-0" />
              <span className="text-base truncate mx-auto">Run Analysis</span>
            </Button>

            <Button
              variant="outline"
              className="w-full h-10 bg-[#1a1a1a] hover:bg-[#252525] text-white border-[#333] flex items-center mt-4"
              onClick={() => setShowSettings(!showSettings)}
            >
              <Settings className="h-4 w-4 mr-2 flex-shrink-0" />
              <span className="text-base truncate mx-auto">Settings</span>
            </Button>
          </div>
        </div>
      </div>

      {/* Status Bar with Console Logs */}
      <div className={`h-10 bg-[#0a0a0a] border-t border-[#222] flex items-center px-4 mt-2 flex-shrink-0 ${isBackendInitializing ? 'pointer-events-none opacity-50' : ''}`}>
        {/* System Stats */}
        <div className="flex items-center space-x-4 text-base text-white">
          <span>{mounted ? formatTime() : "--:--:--"}</span>
          <span>CPU: {cpuUsage}%</span>
          <span>GPU: {gpuUsage}%</span>
          <span>FPS: {currentFps}</span>
          <span>Detection: {detectionTime}ms</span>
        </div>

        {/* Console Logs - Now positioned on the right side */}
        <div className="ml-auto overflow-hidden relative h-full flex items-center">
          {logs.length > 0 && (
            <div className="flex items-center space-x-2">
              {logs.slice(-2).map((log) => (
                <div key={log.id} className="flex items-center shrink-0">
                  {log.type === "info" && <Info className="h-4 w-4 text-blue-400 mr-1 flex-shrink-0" />}
                  {log.type === "success" && <CheckCircle className="h-4 w-4 text-green-400 mr-1 flex-shrink-0" />}
                  {log.type === "error" && <AlertCircle className="h-4 w-4 text-red-400 mr-1 flex-shrink-0" />}
                  {log.type === "warning" && <AlertCircle className="h-4 w-4 text-yellow-400 mr-1 flex-shrink-0" />}
                  <span
                    className={`text-base ${
                      log.type === "info"
                        ? "text-blue-100"
                        : log.type === "success"
                          ? "text-green-100"
                          : log.type === "error"
                            ? "text-red-100"
                            : "text-yellow-100"
                    }`}
                  >
                    {log.message}
                  </span>
                </div>
              ))}
              <div ref={logsEndRef} />
            </div>
          )}
        </div>
      </div>

      {/* Settings Modal */}
      {showSettings && (
        <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50">
          <Card className="w-full max-w-2xl bg-[#1a1a1a] border-[#333]">
            <div className="flex justify-between items-center p-3 border-b border-[#333]">
              <h3 className="text-white text-lg font-medium">Settings</h3>
              <Button variant="ghost" size="sm" onClick={() => setShowSettings(false)} className="text-white">
                ✕
              </Button>
            </div>
            <CardContent className="p-4">
              <Tabs defaultValue="general">
                <TabsList className="bg-[#0a0a0a]">
                  <TabsTrigger value="general">General</TabsTrigger>
                  <TabsTrigger value="detection">Detection</TabsTrigger>
                  <TabsTrigger value="cloud">Cloud Storage</TabsTrigger>
                </TabsList>

                <TabsContent value="general" className="space-y-4 mt-4">
                  <div className="space-y-2">
                    <Label className="text-white text-base">Default Output Path</Label>
                    <div className="flex gap-2">
                      <Input
                        value={defaultOutputPath}
                        onChange={(e) => setDefaultOutputPath(e.target.value)}
                        className="bg-[#0a0a0a] border-[#333] text-white"
                      />
                      <Button variant="outline" className="bg-[#0a0a0a] hover:bg-[#252525] text-white border-[#333]">
                        <Folder className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>

                  <div className="space-y-4">
                    <div className="flex items-center space-x-2">
                      <Checkbox
                        id="organize-date"
                        checked={organizeByDate}
                        onCheckedChange={(checked: boolean) => setOrganizeByDate(checked as boolean)}
                      />
                      <div className="grid gap-1.5">
                        <Label htmlFor="organize-date" className="text-white">
                          <Calendar className="h-4 w-4 inline mr-2" />
                          Organize output by date (e.g., /output/2025-05-05/)
                        </Label>
                      </div>
                    </div>

                    <div className="flex items-center space-x-2">
                      <Checkbox
                        id="auto-delete"
                        checked={autoDeleteRaw}
                        onCheckedChange={(checked: boolean) => setAutoDeleteRaw(checked as boolean)}
                      />
                      <div className="grid gap-1.5">
                        <Label htmlFor="auto-delete" className="text-white">
                          <Trash2 className="h-4 w-4 inline mr-2" />
                          Auto-delete raw images after processing
                        </Label>
                      </div>
                    </div>
                  </div>
                </TabsContent>

                <TabsContent value="detection" className="space-y-4 mt-4">
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <Label className="text-white">
                        <Sliders className="h-4 w-4 inline mr-2" />
                        Confidence Threshold: {confidenceThreshold.toFixed(2)}
                      </Label>
                    </div>
                    <Slider
                      value={[confidenceThreshold]}
                      onValueChange={(value: SetStateAction<number>[]) => setConfidenceThreshold(value[0])}
                      min={0}
                      max={1}
                      step={0.01}
                      className="py-1"
                    />
                  </div>

                  <div className="space-y-2">
                    <Label className="text-white">
                      <Filter className="h-4 w-4 inline mr-2" />
                      Class Filter
                    </Label>

                    <div className="space-y-2">
                      <div className="flex items-center space-x-2">
                        <Checkbox
                          id="linear-cracks"
                          checked={classFilter.includes("linear")}
                          onCheckedChange={(checked: any) => {
                            if (checked) {
                              setClassFilter((prev) => [...prev, "linear"])
                            } else {
                              setClassFilter((prev) => prev.filter((item) => item !== "linear"))
                            }
                          }}
                        />
                        <Label htmlFor="linear-cracks" className="text-white">
                          Linear Cracks
                        </Label>
                      </div>

                      <div className="flex items-center space-x-2">
                        <Checkbox
                          id="alligator-cracks"
                          checked={classFilter.includes("alligator")}
                          onCheckedChange={(checked: any) => {
                            if (checked) {
                              setClassFilter((prev) => [...prev, "alligator"])
                            } else {
                              setClassFilter((prev) => prev.filter((item) => item !== "alligator"))
                            }
                          }}
                        />
                        <Label htmlFor="alligator-cracks" className="text-white">
                          Alligator Cracks
                        </Label>
                      </div>

                      <div className="flex items-center space-x-2">
                        <Checkbox
                          id="potholes"
                          checked={classFilter.includes("pothole")}
                          onCheckedChange={(checked: any) => {
                            if (checked) {
                              setClassFilter((prev) => [...prev, "pothole"])
                            } else {
                              setClassFilter((prev) => prev.filter((item) => item !== "pothole"))
                            }
                          }}
                        />
                        <Label htmlFor="potholes" className="text-white">
                          Potholes
                        </Label>
                      </div>
                    </div>
                  </div>
                </TabsContent>

                <TabsContent value="cloud" className="space-y-4 mt-4">
                  <div className="space-y-2">
                    <Label className="text-white">
                      <Cloud className="h-4 w-4 inline mr-2" />
                      Bucket/Folder Path
                    </Label>
                    <Input
                      value={bucketPath}
                      onChange={(e) => setBucketPath(e.target.value)}
                      className="bg-[#0a0a0a] border-[#333] text-white"
                      placeholder="s3://my-bucket/detections/"
                    />
                  </div>

                  <div className="flex items-center justify-between">
                    <div className="flex items-center">
                      <span className="text-white text-sm mr-2">Status:</span>
                      <span
                        className={`inline-block w-2.5 h-2.5 rounded-full mr-1 ${
                          cloudConnected ? "bg-green-500" : cloudConnecting ? "bg-yellow-500" : "bg-red-500"
                        }`}
                      ></span>
                      <span className="text-white text-sm">
                        {cloudConnected ? "Connected" : cloudConnecting ? "Connecting..." : "Disconnected"}
                      </span>
                    </div>

                    <Button
                      variant="outline"
                      size="sm"
                      className="bg-[#0a0a0a] hover:bg-[#252525] text-white border-[#333]"
                      onClick={testCloudConnection}
                      disabled={cloudConnecting}
                    >
                      {cloudConnecting ? "Testing..." : "Test Connection"}
                    </Button>
                  </div>
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Add loading indicator */}
      {cameraLoading && (
        <div className="absolute inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-[#1a1a1a] p-4 rounded-lg flex flex-col items-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-white mb-2"></div>
            <span className="text-white">Updating camera settings...</span>
          </div>
        </div>
      )}
    </div>
  )
}
