"use client"

import { useState, useEffect, useRef } from "react"
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

type LogType = "info" | "success" | "error" | "warning"

interface LogMessage {
  id: number
  type: LogType
  message: string
  timestamp: string
}

export default function Dashboard() {
  const router = useRouter()
  // Video stream state
  const [isPlaying, setIsPlaying] = useState(false)
  const [brightness, setBrightness] = useState(50)
  const [exposure, setExposure] = useState(50)
  const [cameraDevice, setCameraDevice] = useState("default")
  const [resolution, setResolution] = useState("720p")
  const [fps, setFps] = useState("30")
  const [flipVertical, setFlipVertical] = useState(false)

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
    setIsPlaying(!isPlaying)
    addLog("info", isPlaying ? "Video stream paused" : "Video stream started")
  }

  const toggleDetection = () => {
    if (!detectionActive) {
      addLog("info", "Starting detection...")
      setDetectionActive(true)
      // Simulate detection starting
      simulateDetection()
    } else {
      setDetectionActive(false)
      addLog("info", "Detection stopped")
    }
  }

  const toggleFlipVertical = () => {
    setFlipVertical(!flipVertical)
    addLog("info", flipVertical ? "Video flip disabled" : "Video flipped vertically")
  }

  const toggleGps = () => {
    if (!gpsConnected && !gpsConnecting) {
      setGpsConnecting(true)
      addLog("info", "Connecting to GPS...")

      // Simulate connection delay
      setTimeout(() => {
        setGpsConnecting(false)
        setGpsConnected(true)
        addLog("success", "GPS connected successfully")

        // Simulate GPS data updates
        const interval = setInterval(() => {
          const lat = (40.7128 + Math.random() * 0.01).toFixed(6)
          const lng = (74.006 + Math.random() * 0.01).toFixed(6)
          setGpsData({ longitude: lng, latitude: lat })
        }, 3000)
        return () => clearInterval(interval)
      }, 2000)
    } else if (gpsConnected) {
      setGpsConnected(false)
      addLog("info", "GPS disconnected")
    }
  }

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

  // Simulate detection process
  const simulateDetection = () => {
    if (detectionActive) {
      const interval = setInterval(() => {
        // Update defect counts randomly
        const newLinear = Math.floor(Math.random() * 10)
        const newAlligator = Math.floor(Math.random() * 5)
        const newPotholes = Math.floor(Math.random() * 3)

        // Log if new defects are found
        if (newLinear > linearCracks) {
          addLog("warning", `Detected ${newLinear - linearCracks} new linear crack(s)`)
        }
        if (newAlligator > alligatorCracks) {
          addLog("warning", `Detected ${newAlligator - alligatorCracks} new alligator crack(s)`)
        }
        if (newPotholes > potholes) {
          addLog("warning", `Detected ${newPotholes - potholes} new pothole(s)`)
        }

        setLinearCracks(newLinear)
        setAlligatorCracks(newAlligator)
        setPotholes(newPotholes)

        // Update system stats
        setCpuUsage(Math.floor(20 + Math.random() * 30))
        setGpuUsage(Math.floor(40 + Math.random() * 50))
        setDetectionTime(Math.floor(50 + Math.random() * 100))
        setCurrentFps(Math.floor(25 + Math.random() * 5))

        // Occasionally log system status
        if (Math.random() > 0.9) {
          if (cpuUsage > 40) {
            addLog("warning", `High CPU usage: ${cpuUsage}%`)
          } else {
            addLog("info", `System running normally. CPU: ${cpuUsage}%, GPU: ${gpuUsage}%`)
          }
        }
      }, 1000)
      return () => clearInterval(interval)
    }
  }

  // Initialize simulation
  useEffect(() => {
    let interval: NodeJS.Timeout

    // Add initial logs
    addLog("info", "System initialized")
    addLog("info", "Ready to start detection")

    if (detectionActive) {
      interval = setInterval(() => {
        // Update defect counts randomly
        const newLinear = Math.floor(Math.random() * 10)
        const newAlligator = Math.floor(Math.random() * 5)
        const newPotholes = Math.floor(Math.random() * 3)

        // Log if new defects are found
        if (newLinear > linearCracks) {
          addLog("warning", `Detected ${newLinear - linearCracks} new linear crack(s)`)
        }
        if (newAlligator > alligatorCracks) {
          addLog("warning", `Detected ${newAlligator - alligatorCracks} new alligator crack(s)`)
        }
        if (newPotholes > potholes) {
          addLog("warning", `Detected ${newPotholes - potholes} new pothole(s)`)
        }

        setLinearCracks(newLinear)
        setAlligatorCracks(newAlligator)
        setPotholes(newPotholes)

        // Update system stats
        setCpuUsage(Math.floor(20 + Math.random() * 30))
        setGpuUsage(Math.floor(40 + Math.random() * 50))
        setDetectionTime(Math.floor(50 + Math.random() * 100))
        setCurrentFps(Math.floor(25 + Math.random() * 5))
      }, 1000)
    }

    return () => {
      if (interval) clearInterval(interval)
    }
  }, [detectionActive])

  // Format time as HH:MM:SS
  const formatTime = () => {
    const now = new Date()
    return `${String(now.getHours()).padStart(2, "0")}:${String(now.getMinutes()).padStart(2, "0")}:${String(now.getSeconds()).padStart(2, "0")}`
  }

  return (
    <div className="h-screen flex flex-col bg-[#0a0a0a] overflow-hidden">
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
            {/* Flip Vertical Button */}
            <Button
              variant="ghost"
              size="sm"
              className={`absolute top-2 right-2 h-7 w-7 p-0 rounded-full ${flipVertical ? "bg-[#050e39]/20" : "bg-[#1a1a1a]/50 hover:bg-[#1a1a1a]/70"}`}
              onClick={toggleFlipVertical}
              title="Flip Vertical"
            >
              <FlipVertical className="h-4 w-4 text-white" />
            </Button>

            {!isPlaying && (
              <div className="text-center">
                <Camera className="w-16 h-16 text-white/30 mx-auto mb-2" />
                <p className="text-white/50 text-sm">Press Start Detection to begin</p>
              </div>
            )}
            {isPlaying && detectionActive && (
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="border-2 border-[#050e39] w-20 h-20 rounded-lg flex items-center justify-center">
                  <span className="text-[#050e39] text-xs">DETECTING</span>
                </div>
              </div>
            )}

            {/* Apply flip vertical transformation when active */}
            {flipVertical && (
              <div className="absolute inset-0 bg-[#050e39]/10 flex items-center justify-center pointer-events-none">
                <span className="text-[#050e39] text-xs font-bold">FLIPPED</span>
              </div>
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
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-2 lg:gap-0 mt-2 flex-shrink-0">
        {/* Video Control Panel */}
        <div className="border-r border-[#222] flex flex-col bg-[#0a0a0a]">
          <div className="h-8 bg-[#0a0a0a] flex items-center px-4 border-b border-[#222]">
            <span className="text-base text-white">Video Controls</span>
          </div>
          <div className="p-4 flex-1 flex flex-col justify-between">
            <div className="grid grid-cols-2 gap-2">
              <Select value={cameraDevice} onValueChange={setCameraDevice}>
                <SelectTrigger className="h-9 text-sm bg-[#1a1a1a] border-[#333]">
                  <SelectValue placeholder="Select camera" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="default">Default Camera</SelectItem>
                  <SelectItem value="usb">USB Camera</SelectItem>
                  <SelectItem value="ip">IP Camera</SelectItem>
                </SelectContent>
              </Select>

              <Select value={resolution} onValueChange={setResolution}>
                <SelectTrigger className="h-9 text-sm bg-[#1a1a1a] border-[#333]">
                  <SelectValue placeholder="Resolution" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="480p">480p</SelectItem>
                  <SelectItem value="720p">720p (30fps)</SelectItem>
                  <SelectItem value="1080p">1080p</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2 mt-4">
              <div className="flex justify-between items-center">
                <span className="text-sm text-white">Brightness</span>
                <span className="text-sm text-white">{brightness}%</span>
              </div>
              <Slider
                value={[brightness]}
                onValueChange={(value) => setBrightness(value[0])}
                max={100}
                step={1}
                className="py-1"
              />
            </div>

            <div className="space-y-2 mt-4">
              <div className="flex justify-between items-center">
                <span className="text-sm text-white">Exposure</span>
                <span className="text-sm text-white">{exposure}%</span>
              </div>
              <Slider
                value={[exposure]}
                onValueChange={(value) => setExposure(value[0])}
                max={100}
                step={1}
                className="py-1"
              />
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
      <div className="h-10 bg-[#0a0a0a] border-t border-[#222] flex items-center px-4 mt-2 flex-shrink-0">
        {/* System Stats */}
        <div className="flex items-center space-x-4 text-base text-white">
          <span>{formatTime()}</span>
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
                        onCheckedChange={(checked) => setOrganizeByDate(checked as boolean)}
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
                        onCheckedChange={(checked) => setAutoDeleteRaw(checked as boolean)}
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
                      onValueChange={(value) => setConfidenceThreshold(value[0])}
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
                          onCheckedChange={(checked) => {
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
                          onCheckedChange={(checked) => {
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
                          onCheckedChange={(checked) => {
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
    </div>
  )
}
