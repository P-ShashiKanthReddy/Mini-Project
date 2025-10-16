import { useEffect, useState } from 'react';
import { DetectionMetrics } from '../types';
import { Activity, Clock, Zap, Cpu } from 'lucide-react';

interface MetricsPanelProps {
  currentMetrics: DetectionMetrics | null;
}

export default function MetricsPanel({ currentMetrics }: MetricsPanelProps) {
  const [metrics, setMetrics] = useState<DetectionMetrics | null>(null);

  useEffect(() => {
    if (currentMetrics) {
      setMetrics(currentMetrics);
    }
  }, [currentMetrics]);

  const MetricCard = ({
    icon: Icon,
    label,
    value,
    unit
  }: {
    icon: any;
    label: string;
    value: number | string;
    unit: string;
  }) => (
    <div className="metric-card">
      <div className="metric-icon">
        <Icon size={20} />
      </div>
      <div className="metric-content">
        <span className="metric-label">{label}</span>
        <span className="metric-value">
          {typeof value === 'number' ? value.toFixed(2) : value}
          <span className="metric-unit">{unit}</span>
        </span>
      </div>
    </div>
  );

  return (
    <div className="metrics-panel">
      <h3>Real-time Metrics</h3>
      <div className="metrics-grid">
        <MetricCard
          icon={Clock}
          label="Inference Time"
          value={metrics?.inferenceTime || 0}
          unit="ms"
        />
        <MetricCard
          icon={Zap}
          label="Power Consumption"
          value={metrics?.powerConsumption || 0}
          unit="W"
        />
        <MetricCard
          icon={Cpu}
          label="CPU Utilization"
          value={metrics?.cpuUtilization || 0}
          unit="%"
        />
        <MetricCard
          icon={Activity}
          label="GPU Utilization"
          value={metrics?.gpuUtilization || 0}
          unit="%"
        />
      </div>
    </div>
  );
}
