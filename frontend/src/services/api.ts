import { DetectionMode, DetectionResult } from '../types';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000';

export const api = {
  async startDetection(mode: DetectionMode): Promise<{ success: boolean; message: string }> {
    const response = await fetch(`${API_BASE_URL}/api/detection/start`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ mode }),
    });
    return response.json();
  },

  async stopDetection(): Promise<{ success: boolean; message: string }> {
    const response = await fetch(`${API_BASE_URL}/api/detection/stop`, {
      method: 'POST',
    });
    return response.json();
  },

  async switchMode(mode: DetectionMode): Promise<{ success: boolean; message: string }> {
    const response = await fetch(`${API_BASE_URL}/api/detection/mode`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ mode }),
    });
    return response.json();
  },

  async getFrame(): Promise<DetectionResult> {
    const response = await fetch(`${API_BASE_URL}/api/detection/frame`);
    return response.json();
  },

  async getStatus(): Promise<{ running: boolean; mode: DetectionMode }> {
    const response = await fetch(`${API_BASE_URL}/api/detection/status`);
    return response.json();
  },

  getStreamUrl(): string {
    return `${API_BASE_URL}/api/detection/stream`;
  },
};
