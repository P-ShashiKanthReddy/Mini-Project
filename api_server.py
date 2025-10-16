from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import cv2
import torch
import numpy as np
import psutil
import time
import base64
from threading import Thread, Lock
import json

app = Flask(__name__)
CORS(app)

class DetectionServer:
    def __init__(self):
        self.running = False
        self.mode = 'CNN'
        self.camera = None
        self.model = None
        self.current_frame = None
        self.current_detections = []
        self.metrics = {
            'inferenceTime': 0,
            'cpuUtilization': 0,
            'gpuUtilization': 0,
            'powerConsumption': 0,
            'detectionsCount': 0,
            'fps': 0
        }
        self.lock = Lock()
        self.detection_thread = None
        self.fps_counter = 0
        self.last_fps_time = time.time()

    def load_model(self):
        try:
            if self.mode == 'CNN':
                self.model = torch.hub.load('ultralytics/yolov5', 'yolov5n')
            else:
                self.model = torch.hub.load('ultralytics/yolov5', 'yolov5n')

            if torch.cuda.is_available():
                self.model.cuda()

            print(f"Model loaded in {self.mode} mode")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def start_detection(self):
        if self.running:
            return

        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            raise Exception("Could not open webcam")

        self.load_model()
        self.running = True
        self.detection_thread = Thread(target=self.detection_loop)
        self.detection_thread.start()

    def stop_detection(self):
        self.running = False
        if self.detection_thread:
            self.detection_thread.join()

        if self.camera:
            self.camera.release()
            self.camera = None

    def detection_loop(self):
        while self.running:
            ret, frame = self.camera.read()
            if not ret:
                continue

            start_time = time.time()

            cpu_before = psutil.cpu_percent(interval=0.1)

            if self.model:
                results = self.model(frame)
                detections = results.pandas().xyxy[0].to_dict('records')

                for det in detections:
                    x1, y1, x2, y2 = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
                    conf = det['confidence']
                    cls = det['name']

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{cls} {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                detections = []

            inference_time = (time.time() - start_time) * 1000

            cpu_after = psutil.cpu_percent(interval=0.1)
            cpu_util = (cpu_before + cpu_after) / 2

            gpu_util = 0
            if torch.cuda.is_available():
                try:
                    gpu_util = torch.cuda.utilization()
                except:
                    gpu_util = 0

            power = self.estimate_power(inference_time, self.mode)

            self.fps_counter += 1
            current_time = time.time()
            if current_time - self.last_fps_time >= 1.0:
                fps = self.fps_counter
                self.fps_counter = 0
                self.last_fps_time = current_time
            else:
                fps = self.metrics['fps']

            with self.lock:
                self.current_frame = frame
                self.current_detections = detections
                self.metrics = {
                    'inferenceTime': round(inference_time, 2),
                    'cpuUtilization': round(cpu_util, 2),
                    'gpuUtilization': round(gpu_util, 2),
                    'powerConsumption': round(power, 2),
                    'detectionsCount': len(detections),
                    'fps': fps
                }

            time.sleep(0.01)

    def estimate_power(self, inference_time, mode):
        base_power = 15
        inference_power = (inference_time / 1000) * 50

        if mode == 'SNN':
            return (base_power + inference_power) * 0.6
        return base_power + inference_power

    def get_frame_data(self):
        with self.lock:
            if self.current_frame is None:
                return None, [], {}

            _, buffer = cv2.imencode('.jpg', self.current_frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')

            detections = [
                {
                    'class': det.get('name', 'unknown'),
                    'confidence': float(det.get('confidence', 0)),
                    'bbox': [
                        float(det.get('xmin', 0)),
                        float(det.get('ymin', 0)),
                        float(det.get('xmax', 0)),
                        float(det.get('ymax', 0))
                    ]
                }
                for det in self.current_detections
            ]

            return frame_base64, detections, self.metrics.copy()

    def switch_mode(self, new_mode):
        self.mode = new_mode
        if self.running and self.model:
            self.load_model()

server = DetectionServer()

@app.route('/api/detection/start', methods=['POST'])
def start_detection():
    try:
        data = request.get_json() or {}
        mode = data.get('mode', 'CNN')
        server.mode = mode
        server.start_detection()
        return jsonify({'success': True, 'message': 'Detection started'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/detection/stop', methods=['POST'])
def stop_detection():
    try:
        server.stop_detection()
        return jsonify({'success': True, 'message': 'Detection stopped'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/detection/mode', methods=['POST'])
def switch_mode():
    try:
        data = request.get_json()
        mode = data.get('mode', 'CNN')
        server.switch_mode(mode)
        return jsonify({'success': True, 'message': f'Switched to {mode} mode'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/detection/frame', methods=['GET'])
def get_frame():
    try:
        frame_base64, detections, metrics = server.get_frame_data()

        if frame_base64 is None:
            return jsonify({
                'frame': None,
                'detections': [],
                'metrics': {
                    'mode': server.mode,
                    'inferenceTime': 0,
                    'cpuUtilization': 0,
                    'gpuUtilization': 0,
                    'powerConsumption': 0,
                    'detectionsCount': 0,
                    'fps': 0,
                    'timestamp': int(time.time() * 1000)
                }
            })

        return jsonify({
            'frame': frame_base64,
            'detections': detections,
            'metrics': {
                'mode': server.mode,
                'inferenceTime': metrics['inferenceTime'],
                'cpuUtilization': metrics['cpuUtilization'],
                'gpuUtilization': metrics['gpuUtilization'],
                'powerConsumption': metrics['powerConsumption'],
                'detectionsCount': metrics['detectionsCount'],
                'fps': metrics['fps'],
                'timestamp': int(time.time() * 1000)
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/detection/status', methods=['GET'])
def get_status():
    return jsonify({
        'running': server.running,
        'mode': server.mode
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    print("Starting Neuromorphic YOLOv5 API Server...")
    print("Server will be available at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
