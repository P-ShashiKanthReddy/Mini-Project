import { useEffect, useRef, useState } from 'react';
import { DetectionMode } from '../types';
import { api } from '../services/api';
import { Video, Pause, Play } from 'lucide-react';

interface DetectionViewProps {
  mode: DetectionMode;
  isRunning: boolean;
  onToggleRunning: () => void;
}

export default function DetectionView({ mode, isRunning, onToggleRunning }: DetectionViewProps) {
  const videoRef = useRef<HTMLImageElement>(null);
  const [fps, setFps] = useState(0);
  const [detectionCount, setDetectionCount] = useState(0);

  useEffect(() => {
    if (!isRunning) return;

    let frameCount = 0;
    const fpsInterval = setInterval(() => {
      setFps(frameCount);
      frameCount = 0;
    }, 1000);

    const fetchFrame = async () => {
      try {
        const result = await api.getFrame();
        if (videoRef.current && result.frame) {
          videoRef.current.src = `data:image/jpeg;base64,${result.frame}`;
          frameCount++;
          setDetectionCount(result.detections.length);
        }
      } catch (error) {
        console.error('Error fetching frame:', error);
      }
    };

    const intervalId = setInterval(fetchFrame, 33);

    return () => {
      clearInterval(intervalId);
      clearInterval(fpsInterval);
    };
  }, [isRunning]);

  return (
    <div className="detection-view">
      <div className="video-container">
        {isRunning ? (
          <img
            ref={videoRef}
            alt="Detection feed"
            className="video-feed"
          />
        ) : (
          <div className="video-placeholder">
            <Video size={64} />
            <p>Click Start to begin detection</p>
          </div>
        )}
      </div>

      <div className="video-controls">
        <button
          onClick={onToggleRunning}
          className={`control-button ${isRunning ? 'stop' : 'start'}`}
        >
          {isRunning ? (
            <>
              <Pause size={20} />
              Stop Detection
            </>
          ) : (
            <>
              <Play size={20} />
              Start Detection
            </>
          )}
        </button>

        <div className="video-stats">
          <div className="stat">
            <span className="stat-label">Mode:</span>
            <span className={`stat-value mode-${mode.toLowerCase()}`}>{mode}</span>
          </div>
          <div className="stat">
            <span className="stat-label">FPS:</span>
            <span className="stat-value">{fps}</span>
          </div>
          <div className="stat">
            <span className="stat-label">Detections:</span>
            <span className="stat-value">{detectionCount}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
