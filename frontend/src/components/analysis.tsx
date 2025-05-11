"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ArrowLeft, Filter, ImageIcon, CheckCircle, MapPin, BarChart2, X } from "lucide-react"
import { useRouter } from "next/navigation"

type AnalysisStep = "raw-detections" | "calculate-severity" | "predictive-analysis" | "upload"
type ImageFile = {
  id: string
  name: string
  path: string
  selected?: boolean
}

export default function Analysis() {
  const router = useRouter()
  const [currentStep, setCurrentStep] = useState<AnalysisStep>("raw-detections")
  const [selectedImage, setSelectedImage] = useState<string | null>(null)
  const [calculatingProgress, setCalculatingProgress] = useState(0)
  const [isCalculating, setIsCalculating] = useState(false)

  // New state for predictive analysis
  const [isPredicting, setIsPredicting] = useState(false)
  const [predictingProgress, setPredictingProgress] = useState(0)
  const [showCompletionPopup, setShowCompletionPopup] = useState(false)

  // Mock data for images
  const [images] = useState<ImageFile[]>([
    { id: "1", name: "image1.jpg", path: "/images/1.jpg" },
    { id: "2", name: "image2.jpg", path: "/images/2.jpg" },
    { id: "3", name: "image3.jpg", path: "/images/3.jpg" },
    { id: "4", name: "image4.jpg", path: "/images/4.jpg" },
    { id: "5", name: "image5.jpg", path: "/images/5.jpg" },
    { id: "6", name: "image6.jpg", path: "/images/6.jpg" },
    { id: "7", name: "image7.jpg", path: "/images/7.jpg" },
    { id: "8", name: "image8.jpg", path: "/images/8.jpg" },
    { id: "9", name: "image9.jpg", path: "/images/9.jpg" },
    { id: "10", name: "image10.jpg", path: "/images/10.jpg" },
    { id: "11", name: "image11.jpg", path: "/images/11.jpg" },
    { id: "12", name: "image12.jpg", path: "/images/12.jpg" },
  ])

  const [selectedImages, setSelectedImages] = useState<string[]>([])

  const [linearCracks, setLinearCracks] = useState(Math.floor(Math.random() * 5))
  const [alligatorCracks, setAlligatorCracks] = useState(Math.floor(Math.random() * 3))
  const [potholes, setPotholes] = useState(Math.floor(Math.random() * 2))

  // Add a new state to track completion of each step
  const [stepsCompleted, setStepsCompleted] = useState<Record<AnalysisStep, boolean>>({
    "raw-detections": false,
    "calculate-severity": false,
    "predictive-analysis": false,
    upload: false,
  })

  // Start predictive analysis when entering that step
  useEffect(() => {
    if (currentStep === "predictive-analysis" && !stepsCompleted["predictive-analysis"]) {
      startPredictiveAnalysis()
    }
  }, [currentStep, stepsCompleted])

  // Function to simulate predictive analysis
  const startPredictiveAnalysis = () => {
    setIsPredicting(true)
    setPredictingProgress(0)

    const interval = setInterval(() => {
      setPredictingProgress((prev) => {
        const newProgress = prev + Math.floor(Math.random() * 3) + 1
        if (newProgress >= 100) {
          clearInterval(interval)
          setTimeout(() => {
            setIsPredicting(false)
            setShowCompletionPopup(true)
            setStepsCompleted((prev) => ({
              ...prev,
              "predictive-analysis": true,
            }))
          }, 500)
          return 100
        }
        return newProgress
      })
    }, 150)
  }

  // Update the handleImageSelect function to mark the raw-detections step as completed when at least one image is selected
  const handleImageSelect = (id: string) => {
    if (selectedImages.includes(id)) {
      const newSelectedImages = selectedImages.filter((imgId) => imgId !== id)
      setSelectedImages(newSelectedImages)
      setStepsCompleted((prev) => ({
        ...prev,
        "raw-detections": newSelectedImages.length > 0,
      }))
    } else {
      const newSelectedImages = [...selectedImages, id]
      setSelectedImages(newSelectedImages)
      setStepsCompleted((prev) => ({
        ...prev,
        "raw-detections": true,
      }))
    }
    handleImagePreview(id)
  }

  const handleImagePreview = (id: string) => {
    setSelectedImage(id)
  }

  // Update the handleProceed function to mark steps as completed
  const handleProceed = () => {
    if (currentStep === "raw-detections") {
      setCurrentStep("calculate-severity")
      // Simulate calculation process
      setIsCalculating(true)
      let progress = 0
      const interval = setInterval(() => {
        progress += 1
        setCalculatingProgress(progress)
        if (progress >= 100) {
          clearInterval(interval)
          setIsCalculating(false)
          setStepsCompleted((prev) => ({
            ...prev,
            "calculate-severity": true,
          }))
        }
      }, 50)
    } else if (currentStep === "calculate-severity") {
      setCurrentStep("predictive-analysis")
      // Predictive analysis will start automatically via useEffect
    } else if (currentStep === "predictive-analysis") {
      setCurrentStep("upload")
      // Simulate upload process
      setIsCalculating(true)
      let progress = 0
      const interval = setInterval(() => {
        progress += 2
        setCalculatingProgress(progress)
        if (progress >= 100) {
          clearInterval(interval)
          setIsCalculating(false)
          setStepsCompleted((prev) => ({
            ...prev,
            upload: true,
          }))
        }
      }, 100)
    } else {
      // Return to dashboard after upload
      router.push("/")
    }
  }

  // Add a function to go back to the previous step
  const handlePrevious = () => {
    if (currentStep === "calculate-severity") {
      setCurrentStep("raw-detections")
    } else if (currentStep === "predictive-analysis") {
      setCurrentStep("calculate-severity")
    } else if (currentStep === "upload") {
      setCurrentStep("predictive-analysis")
    }
  }

  // Replace the raw-detections metadata section with a better styled version
  const renderStepContent = () => {
    switch (currentStep) {
      case "raw-detections":
        return (
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-3 h-full">
            <div className="lg:col-span-3 bg-[#1a1a1a] border border-[#333] rounded-md p-3">
              <div className="flex justify-between items-center mb-2">
                <h3 className="text-white text-sm">Files view from output path</h3>
                <Button variant="outline" size="sm" className="h-7 text-xs bg-[#1a1a1a] text-white border-[#333]">
                  <Filter className="h-3 w-3 mr-1" />
                  Filter
                </Button>
              </div>
              <div className="grid grid-cols-4 sm:grid-cols-6 md:grid-cols-8 gap-2">
                {images.map((image) => (
                  <div
                    key={image.id}
                    className={`relative border ${
                      selectedImages.includes(image.id) ? "border-[#050e39]" : "border-[#333]"
                    } rounded-md aspect-square flex items-center justify-center cursor-pointer hover:border-[#050e39] transition-colors`}
                    onClick={() => {
                      handleImageSelect(image.id)
                    }}
                  >
                    <div className="text-center">
                      <ImageIcon className="h-5 w-5 mx-auto text-white/50" />
                      <span className="text-[10px] text-white/70 mt-0.5 block">Image {image.id}</span>
                    </div>
                    {selectedImages.includes(image.id) && (
                      <div className="absolute top-1 right-1 h-2.5 w-2.5 bg-[#050e39] rounded-full"></div>
                    )}
                  </div>
                ))}
              </div>
            </div>
            <div className="bg-[#1a1a1a] border border-[#333] rounded-md p-3">
              <h3 className="text-white text-sm mb-2">Image Preview</h3>
              <div className="aspect-square bg-[#0a0a0a] rounded-md flex items-center justify-center mb-2">
                {selectedImage ? (
                  <div className="text-center">
                    <ImageIcon className="h-8 w-8 mx-auto text-white/30" />
                    <span className="text-xs text-white/50 mt-1 block">Image {selectedImage}</span>
                  </div>
                ) : (
                  <span className="text-white/30 text-xs">Select an image</span>
                )}
              </div>

              {/* Improved metadata display */}
              {selectedImage ? (
                <div className="text-white">
                  <h4 className="text-xs mb-2 border-b border-[#333] pb-1">Metadata Analysis</h4>

                  <div className="bg-[#0a0a0a] rounded-md p-2 mb-2">
                    <div className="flex justify-between items-center mb-1">
                      <span className="text-white/70 text-xs">Total Defects</span>
                      <span className="text-white font-bold text-sm">{Math.floor(Math.random() * 10)}</span>
                    </div>
                    <div className="w-full bg-[#1a1a1a] h-1 rounded-full">
                      <div
                        className="bg-[#050e39] h-full rounded-full"
                        style={{ width: `${Math.floor(Math.random() * 10) * 10}%` }}
                      ></div>
                    </div>
                  </div>

                  <div className="grid grid-cols-3 gap-2 mb-2">
                    <div className="bg-[#0a0a0a] rounded-md p-1.5 text-center">
                      <div className="text-[10px] text-white/50 mb-0.5">Linear</div>
                      <div className="text-sm font-bold">{Math.floor(Math.random() * 5)}</div>
                    </div>
                    <div className="bg-[#0a0a0a] rounded-md p-1.5 text-center">
                      <div className="text-[10px] text-white/50 mb-0.5">Alligator</div>
                      <div className="text-sm font-bold">{Math.floor(Math.random() * 3)}</div>
                    </div>
                    <div className="bg-[#0a0a0a] rounded-md p-1.5 text-center">
                      <div className="text-[10px] text-white/50 mb-0.5">Potholes</div>
                      <div className="text-sm font-bold">{Math.floor(Math.random() * 2)}</div>
                    </div>
                  </div>

                  <div className="bg-[#0a0a0a] rounded-md p-2">
                    <div className="flex items-center mb-1">
                      <MapPin className="h-3 w-3 text-white/50 mr-1" />
                      <span className="text-white/70 text-xs">Location</span>
                    </div>
                    <div className="text-xs font-mono bg-[#1a1a1a] p-1.5 rounded border border-[#333]">
                      40.7128° N, 74.0060° W
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-white/50 text-center p-2 border border-dashed border-[#333] rounded-md text-xs">
                  Select an image to view metadata
                </div>
              )}
            </div>
          </div>
        )
      case "calculate-severity":
        return (
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-3 h-full">
            <div className="lg:col-span-3 bg-[#1a1a1a] border border-[#333] rounded-md p-3">
              <div className="flex justify-between items-center mb-2">
                <h3 className="text-white text-sm">Images With Color Overlays</h3>
                <div className="text-xs text-white/70">{selectedImages.length} images selected</div>
              </div>
              <div className="grid grid-cols-4 sm:grid-cols-6 md:grid-cols-8 gap-2">
                {selectedImages.map((id) => (
                  <div
                    key={id}
                    className="border border-[#333] rounded-md aspect-square flex items-center justify-center cursor-pointer hover:border-[#050e39] transition-colors"
                    onClick={() => handleImagePreview(id)}
                  >
                    <div className="text-center relative">
                      <ImageIcon className="h-5 w-5 mx-auto text-white/50" />
                      <span className="text-[10px] text-white/70 mt-0.5 block">Image {id}</span>
                      {/* Simulated overlay */}
                      <div className="absolute inset-0 bg-red-500/20 rounded-md"></div>
                    </div>
                  </div>
                ))}
              </div>

              {/* Batch severity calculation section */}
              <div className="mt-3 bg-[#0a0a0a] border border-[#333] rounded-md p-2">
                <h3 className="text-white text-xs mb-2">Batch Severity Calculation (All Selected Images)</h3>

                <div className="text-white">
                  <div className="space-y-2">
                    <div>
                      <div className="flex justify-between items-center mb-1">
                        <span className="text-xs text-white/70">Overall Progress</span>
                        <span className="text-xs text-white/70">
                          {isCalculating ? `${calculatingProgress}%` : "100%"}
                        </span>
                      </div>
                      <div className="w-full bg-[#1a1a1a] h-1.5 rounded-full overflow-hidden">
                        <div
                          className="bg-[#050e39] h-full rounded-full transition-all duration-300"
                          style={{ width: `${isCalculating ? calculatingProgress : 100}%` }}
                        ></div>
                      </div>
                    </div>

                    <div className="grid grid-cols-4 gap-2">
                      <div>
                        <div className="flex justify-between items-center mb-0.5">
                          <span className="text-[10px] text-white/70">Preprocessing</span>
                          <span className="text-[10px] text-white/70">
                            {isCalculating && calculatingProgress < 30
                              ? `${Math.min(100, calculatingProgress * 3)}%`
                              : "100%"}
                          </span>
                        </div>
                        <div className="w-full bg-[#1a1a1a] h-1 rounded-full overflow-hidden">
                          <div
                            className="bg-blue-500 h-full rounded-full transition-all duration-300"
                            style={{
                              width: `${isCalculating && calculatingProgress < 30 ? Math.min(100, calculatingProgress * 3) : 100}%`,
                            }}
                          ></div>
                        </div>
                      </div>

                      <div>
                        <div className="flex justify-between items-center mb-0.5">
                          <span className="text-[10px] text-white/70">Grayscale</span>
                          <span className="text-[10px] text-white/70">
                            {isCalculating && calculatingProgress < 50
                              ? `${Math.max(0, Math.min(100, (calculatingProgress - 20) * 3))}%`
                              : "100%"}
                          </span>
                        </div>
                        <div className="w-full bg-[#1a1a1a] h-1 rounded-full overflow-hidden">
                          <div
                            className="bg-green-500 h-full rounded-full transition-all duration-300"
                            style={{
                              width: `${isCalculating && calculatingProgress < 50 ? Math.max(0, Math.min(100, (calculatingProgress - 20) * 3)) : 100}%`,
                            }}
                          ></div>
                        </div>
                      </div>

                      <div>
                        <div className="flex justify-between items-center mb-0.5">
                          <span className="text-[10px] text-white/70">Binarization</span>
                          <span className="text-[10px] text-white/70">
                            {isCalculating && calculatingProgress < 70
                              ? `${Math.max(0, Math.min(100, (calculatingProgress - 40) * 3))}%`
                              : "100%"}
                          </span>
                        </div>
                        <div className="w-full bg-[#1a1a1a] h-1 rounded-full overflow-hidden">
                          <div
                            className="bg-yellow-500 h-full rounded-full transition-all duration-300"
                            style={{
                              width: `${isCalculating && calculatingProgress < 70 ? Math.max(0, Math.min(100, (calculatingProgress - 40) * 3)) : 100}%`,
                            }}
                          ></div>
                        </div>
                      </div>

                      <div>
                        <div className="flex justify-between items-center mb-0.5">
                          <span className="text-[10px] text-white/70">Calculation</span>
                          <span className="text-[10px] text-white/70">
                            {isCalculating && calculatingProgress < 100
                              ? `${Math.max(0, Math.min(100, (calculatingProgress - 60) * 2.5))}%`
                              : "100%"}
                          </span>
                        </div>
                        <div className="w-full bg-[#1a1a1a] h-1 rounded-full overflow-hidden">
                          <div
                            className="bg-red-500 h-full rounded-full transition-all duration-300"
                            style={{
                              width: `${isCalculating && calculatingProgress < 100 ? Math.max(0, Math.min(100, (calculatingProgress - 60) * 2.5)) : 100}%`,
                            }}
                          ></div>
                        </div>
                      </div>
                    </div>

                    <div className="bg-[#1a1a1a] p-2 rounded-md border border-[#333] mt-1">
                      <div className="text-xs font-medium mb-1">Processing Status:</div>
                      <div className="text-xs text-white/70">
                        {isCalculating
                          ? calculatingProgress < 30
                            ? `Applying preprocessing filters to ${selectedImages.length} images...`
                            : calculatingProgress < 50
                              ? `Converting ${selectedImages.length} images to grayscale...`
                              : calculatingProgress < 70
                                ? `Applying binarization algorithms to ${selectedImages.length} images...`
                                : calculatingProgress < 90
                                  ? `Calculating crack lengths and areas across all images...`
                                  : `Finalizing severity scores for ${selectedImages.length} images...`
                          : `All ${selectedImages.length} images processed successfully!`}
                      </div>

                      <div className="grid grid-cols-4 gap-2 mt-2">
                        <div className="bg-[#0a0a0a] p-1.5 rounded-md text-center">
                          <div className="text-[10px] text-white/50 mb-0.5">Images</div>
                          <div className="text-sm font-bold">{selectedImages.length}</div>
                        </div>
                        <div className="bg-[#0a0a0a] p-1.5 rounded-md text-center">
                          <div className="text-[10px] text-white/50 mb-0.5">Defects</div>
                          <div className="text-sm font-bold">{linearCracks + alligatorCracks + potholes}</div>
                        </div>
                        <div className="bg-[#0a0a0a] p-1.5 rounded-md text-center">
                          <div className="text-[10px] text-white/50 mb-0.5">Time</div>
                          <div className="text-sm font-bold">
                            {isCalculating ? "--" : `${Math.floor(Math.random() * 30) + 10}s`}
                          </div>
                        </div>
                        <div className="bg-[#0a0a0a] p-1.5 rounded-md text-center">
                          <div className="text-[10px] text-white/50 mb-0.5">Severity</div>
                          <div className="text-sm font-bold text-yellow-500">{isCalculating ? "--" : "Med"}</div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Image preview section */}
            <div className="bg-[#1a1a1a] border border-[#333] rounded-md p-3">
              <h3 className="text-white text-sm mb-2">Image Preview</h3>
              <div className="aspect-square bg-[#0a0a0a] rounded-md flex items-center justify-center mb-2">
                {selectedImage ? (
                  <div className="text-center relative w-full h-full">
                    <ImageIcon className="h-8 w-8 mx-auto text-white/30 absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2" />
                    <span className="text-xs text-white/50 mt-1 block absolute bottom-4 left-1/2 transform -translate-x-1/2">
                      Image {selectedImage}
                    </span>
                    {/* Simulated overlay */}
                    <div className="absolute inset-0 bg-gradient-to-r from-red-500/20 to-yellow-500/20 rounded-md"></div>
                  </div>
                ) : (
                  <span className="text-white/30 text-xs">Select an image</span>
                )}
              </div>

              {selectedImage ? (
                <div className="text-white">
                  <h4 className="text-xs mb-2 border-b border-[#333] pb-1">Severity Details</h4>

                  <div className="space-y-2">
                    <div className="flex justify-between items-center">
                      <span className="text-white/70 text-xs">Severity Score:</span>
                      <span className="text-yellow-500 font-bold text-sm">Medium (6.4/10)</span>
                    </div>

                    <div className="flex justify-between items-center">
                      <span className="text-white/70 text-xs">Crack Length:</span>
                      <span className="text-white text-sm">42.3 meters</span>
                    </div>

                    <div className="flex justify-between items-center">
                      <span className="text-white/70 text-xs">Affected Area:</span>
                      <span className="text-white text-sm">18.7 m²</span>
                    </div>

                    <div className="flex justify-between items-center">
                      <span className="text-white/70 text-xs">Confidence:</span>
                      <span className="text-white text-sm">92%</span>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-white/50 text-center p-2 border border-dashed border-[#333] rounded-md text-xs">
                  Select an image to view severity details
                </div>
              )}
            </div>
          </div>
        )
      case "predictive-analysis":
        return (
          <>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 h-full">
              <div className="bg-[#1a1a1a] border border-[#333] rounded-md p-4">
                <h3 className="text-white text-base mb-4">Predictive Analysis Results</h3>
                <div className="aspect-video bg-[#0a0a0a] rounded-md flex items-center justify-center mb-4">
                  <span className="text-white/30">Severity Map Visualization</span>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <Card className="bg-[#0a0a0a] border-[#333] p-3">
                    <h4 className="text-white text-sm mb-2">Current Severity</h4>
                    <div className="text-2xl font-bold text-white">Medium</div>
                    <div className="text-xs text-white/50 mt-1">Based on 8 images</div>
                  </Card>
                  <Card className="bg-[#0a0a0a] border-[#333] p-3">
                    <h4 className="text-white text-sm mb-2">Predicted Severity (6 mo)</h4>
                    <div className="text-2xl font-bold text-red-500">High</div>
                    <div className="text-xs text-white/50 mt-1">95% confidence</div>
                  </Card>
                </div>
              </div>
              <div className="bg-[#1a1a1a] border border-[#333] rounded-md p-4">
                <h3 className="text-white text-base mb-4">Maintenance Recommendations</h3>
                <div className="space-y-3">
                  <Card className="bg-[#0a0a0a] border-[#333] p-3">
                    <h4 className="text-white text-sm font-medium">Immediate Action Required</h4>
                    <p className="text-white/70 text-sm mt-1">
                      Seal linear cracks in sections 2, 5, and 7 to prevent water infiltration
                    </p>
                  </Card>
                  <Card className="bg-[#0a0a0a] border-[#333] p-3">
                    <h4 className="text-white text-sm font-medium">Within 3 Months</h4>
                    <p className="text-white/70 text-sm mt-1">
                      Patch potholes in section 3 and repair alligator cracking in section 1
                    </p>
                  </Card>
                  <Card className="bg-[#0a0a0a] border-[#333] p-3">
                    <h4 className="text-white text-sm font-medium">Within 6 Months</h4>
                    <p className="text-white/70 text-sm mt-1">
                      Full resurfacing recommended for sections 4-8 due to predicted deterioration
                    </p>
                  </Card>
                </div>
              </div>
            </div>

            {/* Analysis in progress popup */}
            {isPredicting && (
              <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50">
                <div className="bg-[#1a1a1a] border border-[#333] rounded-md p-6 max-w-md w-full mx-4">
                  <div className="flex items-center justify-center mb-4">
                    <BarChart2 className="h-8 w-8 text-[#050e39] mr-3" />
                    <h3 className="text-white text-lg font-medium">Predictive Analysis in Progress</h3>
                  </div>

                  <div className="space-y-4">
                    <div>
                      <div className="flex justify-between items-center mb-2">
                        <span className="text-sm text-white/70">Overall Progress</span>
                        <span className="text-sm text-white/70">{predictingProgress}%</span>
                      </div>
                      <div className="w-full bg-[#0a0a0a] h-2.5 rounded-full overflow-hidden">
                        <div
                          className="bg-[#050e39] h-full rounded-full transition-all duration-300"
                          style={{ width: `${predictingProgress}%` }}
                        ></div>
                      </div>
                    </div>

                    <div className="text-center text-white/70 text-sm">
                      {predictingProgress < 25
                        ? "Analyzing current defect patterns..."
                        : predictingProgress < 50
                          ? "Calculating deterioration rates..."
                          : predictingProgress < 75
                            ? "Generating future severity predictions..."
                            : "Finalizing maintenance recommendations..."}
                    </div>

                    <div className="grid grid-cols-2 gap-3 mt-2">
                      <div className="bg-[#0a0a0a] p-2 rounded-md text-center">
                        <div className="text-xs text-white/50 mb-1">Time Horizon</div>
                        <div className="text-sm font-bold text-white">6 months</div>
                      </div>
                      <div className="bg-[#0a0a0a] p-2 rounded-md text-center">
                        <div className="text-xs text-white/50 mb-1">Model</div>
                        <div className="text-sm font-bold text-white">ML-RoadDet v2</div>
                      </div>
                    </div>

                    <div className="text-center text-white/50 text-xs">
                      Please wait while we analyze the data and generate predictions...
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Analysis completed popup */}
            {showCompletionPopup && (
              <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50">
                <div className="bg-[#1a1a1a] border border-[#333] rounded-md p-6 max-w-md w-full mx-4">
                  <div className="flex justify-between items-start mb-4">
                    <div className="flex items-center">
                      <CheckCircle className="h-8 w-8 text-green-500 mr-3" />
                      <h3 className="text-white text-lg font-medium">Analysis Complete</h3>
                    </div>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-8 w-8 p-0 text-white/70 hover:text-white"
                      onClick={() => setShowCompletionPopup(false)}
                    >
                      <X className="h-5 w-5" />
                    </Button>
                  </div>

                  <div className="space-y-4">
                    <div className="text-white text-sm">
                      Predictive analysis has been successfully completed. The system has analyzed all defects and
                      generated predictions for future deterioration.
                    </div>

                    <div className="bg-[#0a0a0a] p-3 rounded-md">
                      <h4 className="text-white text-sm font-medium mb-2">Key Findings:</h4>
                      <ul className="space-y-1 text-white/70 text-sm">
                        <li>
                          • Current severity level: <span className="text-yellow-500 font-medium">Medium</span>
                        </li>
                        <li>
                          • Predicted severity (6 mo): <span className="text-red-500 font-medium">High</span>
                        </li>
                        <li>
                          • Confidence level: <span className="text-white font-medium">95%</span>
                        </li>
                        <li>
                          • Priority sections: <span className="text-white font-medium">2, 5, 7</span>
                        </li>
                      </ul>
                    </div>

                    <Button
                      className="w-full bg-[#050e39] hover:bg-[#0a1a5a] text-white"
                      onClick={() => setShowCompletionPopup(false)}
                    >
                      View Detailed Results
                    </Button>
                  </div>
                </div>
              </div>
            )}
          </>
        )
      case "upload":
        return (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 h-full">
            <div className="bg-[#1a1a1a] border border-[#333] rounded-md p-4">
              <h3 className="text-white text-base mb-4">Upload Analysis Results</h3>
              <div className="space-y-4">
                <Card className="bg-[#0a0a0a] border-[#333] p-4">
                  <h4 className="text-white text-sm mb-3">Cloud Storage Upload</h4>
                  <div className="space-y-4">
                    <div className="w-full bg-[#1a1a1a] h-2 rounded-full overflow-hidden">
                      <div
                        className="bg-[#050e39] h-full rounded-full"
                        style={{ width: `${isCalculating ? calculatingProgress : 100}%` }}
                      ></div>
                    </div>
                    <p className="text-white/70 text-sm">
                      {isCalculating
                        ? `Uploading severity images to cloud storage (${calculatingProgress}%)`
                        : "Severity images uploaded successfully"}
                    </p>
                  </div>
                </Card>

                <Card className="bg-[#0a0a0a] border-[#333] p-4">
                  <h4 className="text-white text-sm mb-3">Report Generation</h4>
                  <div className="space-y-2">
                    <div className="flex items-center">
                      <CheckCircle className="h-4 w-4 text-green-500 mr-2" />
                      <span className="text-white text-sm">Severity overlays included</span>
                    </div>
                    <div className="flex items-center">
                      <CheckCircle className="h-4 w-4 text-green-500 mr-2" />
                      <span className="text-white text-sm">Predictive analysis included</span>
                    </div>
                    <div className="flex items-center">
                      <CheckCircle className="h-4 w-4 text-green-500 mr-2" />
                      <span className="text-white text-sm">Maintenance recommendations included</span>
                    </div>
                    <div className="flex items-center">
                      <CheckCircle className="h-4 w-4 text-green-500 mr-2" />
                      <span className="text-white text-sm">Metadata included</span>
                    </div>
                  </div>
                </Card>

                <Card className="bg-[#0a0a0a] border-[#333] p-4">
                  <h4 className="text-white text-sm mb-3">Metadata Transmission</h4>
                  <div className="space-y-2">
                    <div className="flex justify-between items-center">
                      <span className="text-white text-sm">GPS Coordinates</span>
                      <CheckCircle className="h-4 w-4 text-green-500" />
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-white text-sm">Defect Counts</span>
                      <CheckCircle className="h-4 w-4 text-green-500" />
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-white text-sm">Severity Scores</span>
                      <CheckCircle className="h-4 w-4 text-green-500" />
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-white text-sm">Timestamp Data</span>
                      <CheckCircle className="h-4 w-4 text-green-500" />
                    </div>
                  </div>
                </Card>
              </div>
            </div>
            <div className="bg-[#1a1a1a] border border-[#333] rounded-md p-4">
              <h3 className="text-white text-base mb-4">Report Preview</h3>
              <div className="aspect-[3/4] bg-[#0a0a0a] rounded-md flex items-center justify-center mb-4 relative overflow-hidden">
                <div className="absolute inset-0 p-6">
                  <div className="text-center">
                    <h3 className="text-lg font-bold text-white mb-2">Road Defect Analysis Report</h3>
                    <p className="text-sm text-white/70 mb-4">Generated on {new Date().toLocaleDateString()}</p>
                    <div className="w-16 h-1 bg-[#050e39] mx-auto mb-4"></div>
                  </div>

                  <div className="mt-6 space-y-4">
                    <div className="bg-[#1a1a1a] p-2 rounded">
                      <h4 className="text-white text-sm font-medium mb-1">Severity Analysis</h4>
                      <p className="text-sm text-white/70">8 road sections analyzed</p>
                      <div className="h-4 bg-gradient-to-r from-green-500 via-yellow-500 to-red-500 rounded mt-2"></div>
                    </div>

                    <div className="bg-[#1a1a1a] p-2 rounded">
                      <h4 className="text-white text-sm font-medium mb-1">Defect Summary</h4>
                      <div className="grid grid-cols-3 gap-2 text-center">
                        <div>
                          <p className="text-white text-xs">Linear Cracks</p>
                          <p className="text-white text-lg font-bold">{linearCracks}</p>
                        </div>
                        <div>
                          <p className="text-white text-xs">Alligator Cracks</p>
                          <p className="text-white text-lg font-bold">{alligatorCracks}</p>
                        </div>
                        <div>
                          <p className="text-white text-xs">Potholes</p>
                          <p className="text-white text-lg font-bold">{potholes}</p>
                        </div>
                      </div>
                    </div>

                    <div className="bg-[#1a1a1a] p-2 rounded">
                      <h4 className="text-white text-sm font-medium mb-1">Maintenance Priority</h4>
                      <div className="flex items-center justify-between">
                        <span className="text-white text-xs">Current Severity:</span>
                        <span className="text-yellow-500 font-bold text-xs">Medium</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-white text-xs">Predicted (6 mo):</span>
                        <span className="text-red-500 font-bold text-xs">High</span>
                      </div>
                    </div>
                  </div>

                  <div className="absolute bottom-6 left-0 right-0 text-center">
                    <p className="text-xs text-white/50">
                      PDF Report will include detailed severity maps and recommendations
                    </p>
                  </div>
                </div>
              </div>

              <div className="flex justify-between items-center">
                <div className="flex items-center">
                  <div className="w-3 h-3 rounded-full bg-green-500 mr-2"></div>
                  <span className="text-white text-sm">PDF Ready for Download</span>
                </div>
                <Button
                  variant="outline"
                  className="bg-[#050e39] hover:bg-[#0a1a5a] text-white border-0"
                  onClick={() => {}}
                >
                  Preview PDF
                </Button>
              </div>
            </div>
          </div>
        )
      default:
        return null
    }
  }

  return (
    <div className="h-screen flex flex-col bg-[#0a0a0a] overflow-hidden">
      {/* Header with back button and steps */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between p-4 border-b border-[#222]">
        <div className="flex items-center mb-4 sm:mb-0">
          <Button variant="ghost" size="sm" className="h-8 w-8 p-0 mr-2 text-white" onClick={() => router.push("/")}>
            <ArrowLeft className="h-5 w-5" />
          </Button>
          <h1 className="text-xl text-white">Run Analysis</h1>
        </div>

        <Tabs value={currentStep} className="w-full sm:w-auto">
          <TabsList className="bg-[#1a1a1a] w-full sm:w-auto grid grid-cols-4 sm:flex">
            <TabsTrigger
              value="raw-detections"
              className="data-[state=active]:bg-[#050e39] data-[state=active]:text-white"
              disabled={false} // Always allow going to first step
            >
              <div className="flex flex-col sm:flex-row items-center">
                <span className="text-xs sm:mr-2">Step 1</span>
                <span className="text-xs sm:text-sm">Raw Detections</span>
              </div>
            </TabsTrigger>
            <TabsTrigger
              value="calculate-severity"
              className="data-[state=active]:bg-[#050e39] data-[state=active]:text-white"
              disabled={!stepsCompleted["raw-detections"]} // Only enable if previous step is completed
            >
              <div className="flex flex-col sm:flex-row items-center">
                <span className="text-xs sm:mr-2">Step 2</span>
                <span className="text-xs sm:text-sm">Calculate Severity</span>
              </div>
            </TabsTrigger>
            <TabsTrigger
              value="predictive-analysis"
              className="data-[state=active]:bg-[#050e39] data-[state=active]:text-white"
              disabled={!stepsCompleted["calculate-severity"]} // Only enable if previous step is completed
            >
              <div className="flex flex-col sm:flex-row items-center">
                <span className="text-xs sm:mr-2">Step 3</span>
                <span className="text-xs sm:text-sm">Predictive Analysis</span>
              </div>
            </TabsTrigger>
            <TabsTrigger
              value="upload"
              className="data-[state=active]:bg-[#050e39] data-[state=active]:text-white"
              disabled={!stepsCompleted["predictive-analysis"]} // Only enable if previous step is completed
            >
              <div className="flex flex-col sm:flex-row items-center">
                <span className="text-xs sm:mr-2">Step 4</span>
                <span className="text-xs sm:text-sm">Upload</span>
              </div>
            </TabsTrigger>
          </TabsList>
        </Tabs>
      </div>

      {/* Main content */}
      <div className="flex-grow p-3 overflow-hidden">{renderStepContent()}</div>

      {/* Footer with previous and proceed buttons */}
      <div className="p-4 border-t border-[#222] flex justify-between">
        {/* Previous button - only show if not on first step */}
        {currentStep !== "raw-detections" ? (
          <Button
            variant="outline"
            className="bg-[#1a1a1a] hover:bg-[#252525] text-white border-[#333] px-8"
            onClick={handlePrevious}
          >
            Previous
          </Button>
        ) : (
          <div></div> // Empty div to maintain flex layout
        )}

        {/* Proceed button - disabled if current step is not completed */}
        <Button
          className="bg-[#050e39] hover:bg-[#0a1a5a] text-white px-8"
          onClick={handleProceed}
          disabled={
            (currentStep === "raw-detections" && !stepsCompleted["raw-detections"]) ||
            (currentStep === "calculate-severity" && isCalculating) ||
            (currentStep === "predictive-analysis" && !stepsCompleted["predictive-analysis"]) ||
            (currentStep === "upload" && isCalculating)
          }
        >
          {currentStep === "upload" ? "Finish" : "Proceed"}
        </Button>
      </div>
    </div>
  )
}
