import axios from 'axios';

// Custom type definitions for axios
type AxiosRequestConfig = {
  timeout?: number;
  signal?: AbortSignal;
  [key: string]: any;
};

type AxiosError = {
  code?: string;
  message: string;
  isAxiosError?: boolean;
};

const API_BASE_URL = 'http://localhost:5000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Helper function to convert image data to base64
const imageToBase64 = (image: string): string => {
  // If image is already base64, return it
  if (image.startsWith('data:image')) {
    return image;
  }
  // Otherwise, assume it's a raw image data URL and convert
  return `data:image/jpeg;base64,${image}`;
};

// Add type definitions
export interface VideoSettings {
  device: string;
  resolution: string;
  fps: number;
  brightness: number;
  exposure: number;
  flip_vertical: boolean;
  zoom: number;
}

export interface CaptureResponse {
  image: string;
}

export const detectionApi = {
  // Send frame and GPS data for detection
  detect: async (frame: string, gpsData: { longitude: string; latitude: string }) => {
    try {
      const response = await api.post('/detection', {
        frame: imageToBase64(frame),
        gps_data: gpsData
      });
      return response.data;
    } catch (error) {
      console.error('Detection API error:', error);
      throw error;
    }
  },

  // Upload detection results
  upload: async (image: string, defectCounts: { linear: number; alligator: number; pothole: number }, frameCounts: number) => {
    try {
      const response = await api.post('/upload', {
        image: imageToBase64(image),
        defect_counts: defectCounts,
        frame_counts: frameCounts
      });
      return response.data;
    } catch (error) {
      console.error('Upload API error:', error);
      throw error;
    }
  },

  // Capture image from camera
  capture: async (settings: VideoSettings, options?: { signal?: AbortSignal }) => {
    try {
      const config: AxiosRequestConfig = {
        timeout: 5000, // 5 second timeout
        ...(options?.signal && { signal: options.signal })
      };
      
      const response = await axios.post<CaptureResponse>('http://localhost:5000/capture', settings, config);
      return response.data;
    } catch (error) {
      // Check if error is an Axios error
      if (error && typeof error === 'object' && 'isAxiosError' in error) {
        const axiosError = error as AxiosError;
        if (axiosError.code === 'ECONNABORTED') {
          throw new Error('Request timed out');
        }
        if (axiosError.code === 'ERR_CANCELED') {
          throw new Error('Request cancelled');
        }
        throw new Error(axiosError.message);
      }
      throw error;
    }
  },

  // Get GPS data
  getGpsData: async () => {
    try {
      const response = await api.get('/gps');
      return response.data;
    } catch (error) {
      console.error('GPS API error:', error);
      throw error;
    }
  }
}; 