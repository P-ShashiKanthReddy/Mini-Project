import { useEffect, useState } from 'react';
import { supabase } from '../lib/supabase';
import { TrendingDown, TrendingUp } from 'lucide-react';

interface Summary {
  cnn: {
    avgInferenceTime: number;
    avgPower: number;
    count: number;
  };
  snn: {
    avgInferenceTime: number;
    avgPower: number;
    count: number;
  };
}

export default function ComparisonSummary() {
  const [summary, setSummary] = useState<Summary | null>(null);

  useEffect(() => {
    loadSummary();

    const channel = supabase
      .channel('summary_changes')
      .on(
        'postgres_changes',
        {
          event: 'INSERT',
          schema: 'public',
          table: 'detection_metrics'
        },
        () => {
          loadSummary();
        }
      )
      .subscribe();

    return () => {
      supabase.removeChannel(channel);
    };
  }, []);

  const loadSummary = async () => {
    const { data: cnnData } = await supabase
      .from('detection_metrics')
      .select('inference_time, power_consumption')
      .eq('mode', 'CNN');

    const { data: snnData } = await supabase
      .from('detection_metrics')
      .select('inference_time, power_consumption')
      .eq('mode', 'SNN');

    if (cnnData && snnData) {
      const cnnAvgInference = cnnData.reduce((sum, m) => sum + m.inference_time, 0) / cnnData.length || 0;
      const cnnAvgPower = cnnData.reduce((sum, m) => sum + m.power_consumption, 0) / cnnData.length || 0;
      const snnAvgInference = snnData.reduce((sum, m) => sum + m.inference_time, 0) / snnData.length || 0;
      const snnAvgPower = snnData.reduce((sum, m) => sum + m.power_consumption, 0) / snnData.length || 0;

      setSummary({
        cnn: {
          avgInferenceTime: cnnAvgInference,
          avgPower: cnnAvgPower,
          count: cnnData.length
        },
        snn: {
          avgInferenceTime: snnAvgInference,
          avgPower: snnAvgPower,
          count: snnData.length
        }
      });
    }
  };

  if (!summary) {
    return (
      <div className="comparison-summary">
        <h3>Comparison Summary</h3>
        <p className="no-data">No data available yet. Start detection to collect metrics.</p>
      </div>
    );
  }

  const inferenceDiff = summary.cnn.avgInferenceTime > 0
    ? ((summary.snn.avgInferenceTime - summary.cnn.avgInferenceTime) / summary.cnn.avgInferenceTime * 100)
    : 0;

  const powerDiff = summary.cnn.avgPower > 0
    ? ((summary.snn.avgPower - summary.cnn.avgPower) / summary.cnn.avgPower * 100)
    : 0;

  const ComparisonRow = ({
    label,
    cnnValue,
    snnValue,
    unit,
    diff
  }: {
    label: string;
    cnnValue: number;
    snnValue: number;
    unit: string;
    diff: number;
  }) => (
    <div className="comparison-row">
      <div className="comparison-label">{label}</div>
      <div className="comparison-values">
        <div className="value-box cnn">
          <span className="value-mode">CNN</span>
          <span className="value-number">{cnnValue.toFixed(2)} {unit}</span>
        </div>
        <div className="value-box snn">
          <span className="value-mode">SNN</span>
          <span className="value-number">{snnValue.toFixed(2)} {unit}</span>
        </div>
        <div className={`value-diff ${diff < 0 ? 'positive' : 'negative'}`}>
          {diff < 0 ? <TrendingDown size={16} /> : <TrendingUp size={16} />}
          {Math.abs(diff).toFixed(1)}%
        </div>
      </div>
    </div>
  );

  return (
    <div className="comparison-summary">
      <h3>Comparison Summary</h3>
      <ComparisonRow
        label="Avg Inference Time"
        cnnValue={summary.cnn.avgInferenceTime}
        snnValue={summary.snn.avgInferenceTime}
        unit="ms"
        diff={inferenceDiff}
      />
      <ComparisonRow
        label="Avg Power Consumption"
        cnnValue={summary.cnn.avgPower}
        snnValue={summary.snn.avgPower}
        unit="W"
        diff={powerDiff}
      />
      <div className="summary-note">
        <p>
          <strong>Samples:</strong> CNN: {summary.cnn.count} | SNN: {summary.snn.count}
        </p>
      </div>
    </div>
  );
}
