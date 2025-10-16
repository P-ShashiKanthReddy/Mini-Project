/*
  # Create detection metrics table

  1. New Tables
    - `detection_metrics`
      - `id` (uuid, primary key) - Unique identifier for each metric entry
      - `mode` (text) - Detection mode: 'CNN' or 'SNN'
      - `inference_time` (real) - Inference time in milliseconds
      - `cpu_utilization` (real) - CPU utilization percentage
      - `gpu_utilization` (real) - GPU utilization percentage
      - `power_consumption` (real) - Power consumption in watts
      - `detections_count` (integer) - Number of detections in frame
      - `fps` (real) - Frames per second
      - `created_at` (timestamptz) - Timestamp of metric collection

  2. Security
    - Enable RLS on `detection_metrics` table
    - Add policy for public read access (metrics are not sensitive)
    - Add policy for public insert access (allow saving metrics)

  3. Indexes
    - Index on `mode` for efficient filtering
    - Index on `created_at` for time-based queries
*/

CREATE TABLE IF NOT EXISTS detection_metrics (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  mode text NOT NULL CHECK (mode IN ('CNN', 'SNN')),
  inference_time real NOT NULL DEFAULT 0,
  cpu_utilization real NOT NULL DEFAULT 0,
  gpu_utilization real NOT NULL DEFAULT 0,
  power_consumption real NOT NULL DEFAULT 0,
  detections_count integer NOT NULL DEFAULT 0,
  fps real NOT NULL DEFAULT 0,
  created_at timestamptz DEFAULT now()
);

ALTER TABLE detection_metrics ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Allow public read access to metrics"
  ON detection_metrics
  FOR SELECT
  USING (true);

CREATE POLICY "Allow public insert access to metrics"
  ON detection_metrics
  FOR INSERT
  WITH CHECK (true);

CREATE INDEX IF NOT EXISTS idx_detection_metrics_mode 
  ON detection_metrics(mode);

CREATE INDEX IF NOT EXISTS idx_detection_metrics_created_at 
  ON detection_metrics(created_at DESC);
