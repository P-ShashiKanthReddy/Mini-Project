export type DetectionMode = 'CNN' | 'SNN';

export interface DetectionMetrics {
  id?: string;
  timestamp: number;
  mode: DetectionMode;
  inferenceTime: number;
  cpuUtilization: number;
  gpuUtilization: number;
  powerConsumption: number;
  detectionsCount: number;
  fps: number;
  created_at?: string;
}

export interface Detection {
  class: string;
  confidence: number;
  bbox: [number, number, number, number];
}

export interface DetectionResult {
  frame: string;
  detections: Detection[];
  metrics: DetectionMetrics;
}

export interface MetricsSummary {
  cnn: {
    avgInferenceTime: number;
    avgPower: number;
    avgCpuUtil: number;
    avgGpuUtil: number;
    totalDetections: number;
  };
  snn: {
    avgInferenceTime: number;
    avgPower: number;
    avgCpuUtil: number;
    avgGpuUtil: number;
    totalDetections: number;
  };
}
