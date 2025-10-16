import { useEffect, useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { DetectionMetrics, DetectionMode } from '../types';
import { supabase } from '../lib/supabase';

interface ChartData {
  timestamp: string;
  cnnInference?: number;
  snnInference?: number;
  cnnPower?: number;
  snnPower?: number;
}

export default function MetricsChart() {
  const [chartData, setChartData] = useState<ChartData[]>([]);
  const [selectedMetric, setSelectedMetric] = useState<'inference' | 'power'>('inference');

  useEffect(() => {
    loadMetrics();

    const channel = supabase
      .channel('metrics_changes')
      .on(
        'postgres_changes',
        {
          event: 'INSERT',
          schema: 'public',
          table: 'detection_metrics'
        },
        () => {
          loadMetrics();
        }
      )
      .subscribe();

    return () => {
      supabase.removeChannel(channel);
    };
  }, []);

  const loadMetrics = async () => {
    const { data, error } = await supabase
      .from('detection_metrics')
      .select('*')
      .order('created_at', { ascending: true })
      .limit(50);

    if (error) {
      console.error('Error loading metrics:', error);
      return;
    }

    if (data) {
      const groupedData = data.reduce((acc: Record<string, ChartData>, metric: any) => {
        const time = new Date(metric.created_at).toLocaleTimeString();

        if (!acc[time]) {
          acc[time] = { timestamp: time };
        }

        if (metric.mode === 'CNN') {
          acc[time].cnnInference = metric.inference_time;
          acc[time].cnnPower = metric.power_consumption;
        } else {
          acc[time].snnInference = metric.inference_time;
          acc[time].snnPower = metric.power_consumption;
        }

        return acc;
      }, {});

      setChartData(Object.values(groupedData));
    }
  };

  return (
    <div className="metrics-chart">
      <div className="chart-header">
        <h3>Performance Comparison</h3>
        <div className="chart-toggle">
          <button
            className={selectedMetric === 'inference' ? 'active' : ''}
            onClick={() => setSelectedMetric('inference')}
          >
            Inference Time
          </button>
          <button
            className={selectedMetric === 'power' ? 'active' : ''}
            onClick={() => setSelectedMetric('power')}
          >
            Power Consumption
          </button>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#333" />
          <XAxis
            dataKey="timestamp"
            stroke="#999"
            style={{ fontSize: '12px' }}
          />
          <YAxis
            stroke="#999"
            style={{ fontSize: '12px' }}
            label={{
              value: selectedMetric === 'inference' ? 'Time (ms)' : 'Power (W)',
              angle: -90,
              position: 'insideLeft'
            }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: '#1a1a1a',
              border: '1px solid #333',
              borderRadius: '8px'
            }}
          />
          <Legend />
          {selectedMetric === 'inference' ? (
            <>
              <Line
                type="monotone"
                dataKey="cnnInference"
                stroke="#3b82f6"
                name="CNN"
                strokeWidth={2}
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="snnInference"
                stroke="#10b981"
                name="SNN"
                strokeWidth={2}
                dot={false}
              />
            </>
          ) : (
            <>
              <Line
                type="monotone"
                dataKey="cnnPower"
                stroke="#3b82f6"
                name="CNN"
                strokeWidth={2}
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="snnPower"
                stroke="#10b981"
                name="SNN"
                strokeWidth={2}
                dot={false}
              />
            </>
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
