import { useState, useEffect } from 'react';
import { DetectionMode, DetectionMetrics } from './types';
import { api } from './services/api';
import DetectionView from './components/DetectionView';
import ModeSelector from './components/ModeSelector';
import MetricsPanel from './components/MetricsPanel';
import MetricsChart from './components/MetricsChart';
import ComparisonSummary from './components/ComparisonSummary';
import { supabase } from './lib/supabase';
import './App.css';

function App() {
  const [mode, setMode] = useState<DetectionMode>('CNN');
  const [isRunning, setIsRunning] = useState(false);
  const [currentMetrics, setCurrentMetrics] = useState<DetectionMetrics | null>(null);
  const [activeTab, setActiveTab] = useState<'live' | 'analytics'>('live');

  useEffect(() => {
    checkStatus();
  }, []);

  const checkStatus = async () => {
    try {
      const status = await api.getStatus();
      setIsRunning(status.running);
      setMode(status.mode);
    } catch (error) {
      console.error('Error checking status:', error);
    }
  };

  const handleToggleRunning = async () => {
    try {
      if (isRunning) {
        await api.stopDetection();
        setIsRunning(false);
      } else {
        await api.startDetection(mode);
        setIsRunning(true);
        startMetricsCollection();
      }
    } catch (error) {
      console.error('Error toggling detection:', error);
    }
  };

  const handleModeChange = async (newMode: DetectionMode) => {
    try {
      if (isRunning) {
        await api.switchMode(newMode);
      }
      setMode(newMode);
    } catch (error) {
      console.error('Error switching mode:', error);
    }
  };

  const startMetricsCollection = () => {
    const interval = setInterval(async () => {
      try {
        const result = await api.getFrame();
        if (result.metrics) {
          setCurrentMetrics(result.metrics);
          await saveMetrics(result.metrics);
        }
      } catch (error) {
        console.error('Error collecting metrics:', error);
      }
    }, 1000);

    return () => clearInterval(interval);
  };

  const saveMetrics = async (metrics: DetectionMetrics) => {
    try {
      await supabase.from('detection_metrics').insert({
        mode: metrics.mode,
        inference_time: metrics.inferenceTime,
        cpu_utilization: metrics.cpuUtilization,
        gpu_utilization: metrics.gpuUtilization,
        power_consumption: metrics.powerConsumption,
        detections_count: metrics.detectionsCount,
        fps: metrics.fps
      });
    } catch (error) {
      console.error('Error saving metrics:', error);
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-content">
          <div className="logo">
            <div className="logo-icon">⚡</div>
            <div className="logo-text">
              <h1>Neuromorphic YOLOv5</h1>
              <p>Energy-Efficient Object Detection</p>
            </div>
          </div>
          <nav className="header-nav">
            <button
              className={`nav-button ${activeTab === 'live' ? 'active' : ''}`}
              onClick={() => setActiveTab('live')}
            >
              Live Detection
            </button>
            <button
              className={`nav-button ${activeTab === 'analytics' ? 'active' : ''}`}
              onClick={() => setActiveTab('analytics')}
            >
              Analytics
            </button>
          </nav>
        </div>
      </header>

      <main className="app-main">
        {activeTab === 'live' ? (
          <div className="live-view">
            <div className="main-content">
              <DetectionView
                mode={mode}
                isRunning={isRunning}
                onToggleRunning={handleToggleRunning}
              />
              <MetricsPanel currentMetrics={currentMetrics} />
            </div>

            <aside className="sidebar">
              <ModeSelector
                mode={mode}
                onModeChange={handleModeChange}
                disabled={isRunning}
              />

              <div className="info-panel">
                <h3>How It Works</h3>
                <ul>
                  <li>
                    <strong>CNN Mode:</strong> Standard YOLOv5 with high accuracy and power consumption
                  </li>
                  <li>
                    <strong>SNN Mode:</strong> Spiking Neural Network with reduced power and energy efficiency
                  </li>
                  <li>
                    Toggle between modes to compare performance in real-time
                  </li>
                  <li>
                    Metrics are saved automatically for analysis
                  </li>
                </ul>
              </div>
            </aside>
          </div>
        ) : (
          <div className="analytics-view">
            <MetricsChart />
            <ComparisonSummary />
          </div>
        )}
      </main>

      <footer className="app-footer">
        <p>Built with PyTorch, snnTorch, and React | Neuromorphic Computing Research</p>
      </footer>
    </div>
  );
}

export default App;
