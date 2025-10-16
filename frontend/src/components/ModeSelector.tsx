import { DetectionMode } from '../types';
import { Cpu, Zap } from 'lucide-react';

interface ModeSelectorProps {
  mode: DetectionMode;
  onModeChange: (mode: DetectionMode) => void;
  disabled: boolean;
}

export default function ModeSelector({ mode, onModeChange, disabled }: ModeSelectorProps) {
  return (
    <div className="mode-selector">
      <h3>Detection Mode</h3>
      <div className="mode-buttons">
        <button
          className={`mode-button ${mode === 'CNN' ? 'active' : ''}`}
          onClick={() => onModeChange('CNN')}
          disabled={disabled}
        >
          <Cpu size={24} />
          <div className="mode-info">
            <span className="mode-name">CNN Mode</span>
            <span className="mode-desc">Standard YOLOv5</span>
          </div>
        </button>

        <button
          className={`mode-button ${mode === 'SNN' ? 'active' : ''}`}
          onClick={() => onModeChange('SNN')}
          disabled={disabled}
        >
          <Zap size={24} />
          <div className="mode-info">
            <span className="mode-name">SNN Mode</span>
            <span className="mode-desc">Neuromorphic (Low Power)</span>
          </div>
        </button>
      </div>
    </div>
  );
}
