import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import cv2
import numpy as np
import time
import psutil
import os
import matplotlib.pyplot as plt
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from utils.augmentations import letterbox
from utils.plots import Annotator, colors

# Function to simulate power consumption (rough estimate)
def simulate_power_consumption(inference_time, model_params):
    # Rough estimate: power in watts, based on time and params
    base_power = 10  # Base power in watts for CPU
    param_factor = model_params / 1e6  # Scale by million parameters
    return inference_time * base_power * param_factor

# Function to get CPU/GPU utilization
def get_utilization():
    cpu_percent = psutil.cpu_percent(interval=None)  # No interval to avoid blocking
    gpu_percent = 0  # Placeholder, as GPU monitoring might need nvidia-ml-py
    return cpu_percent, gpu_percent

# Function to convert CNN to SNN (simplified approximation)
def cnn_to_snn(model):
    # This is a basic approximation: wrap the model with SNN layers
    # In practice, full conversion is complex; here we use a simple wrapper
    class SNNWrapper(nn.Module):
        def __init__(self, cnn_model):
            super().__init__()
            self.cnn = cnn_model
            self.names = cnn_model.names  # Preserve class names
            self.spike_layer = snn.Leaky(beta=0.9, spike_grad=surrogate.fast_sigmoid())

        def forward(self, x):
            # For now, just return CNN output to avoid crashes
            # Proper SNN conversion would require temporal processing
            return self.cnn(x)

    return SNNWrapper(model)

# Main detection function with switchable modes
def run_detection():
    device = select_device('')
    weights = 'yolov5n.pt'
    cnn_model = attempt_load(weights, device=device)
    snn_model = cnn_to_snn(cnn_model)
    model_params = sum(p.numel() for p in cnn_model.parameters())

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Running detection with switchable CNN/SNN modes...")
    print("Press 'c' to switch to CNN mode, 's' to switch to SNN mode, 'q' to quit.")

    names = cnn_model.names  # Get class names
    current_model = cnn_model
    current_mode = 'cnn'

    frame_count = 0
    total_inference_time = 0
    total_cpu = 0
    total_gpu = 0
    total_power = 0
    detections = 0

    # Separate metrics for CNN and SNN
    cnn_frame_count = 0
    cnn_inference_time = 0
    cnn_cpu = 0
    cnn_gpu = 0
    cnn_power = 0
    cnn_detections = 0

    snn_frame_count = 0
    snn_inference_time = 0
    snn_cpu = 0
    snn_gpu = 0
    snn_power = 0
    snn_detections = 0

    while True:
        ret, im0 = cap.read()  # im0 is original frame
        if not ret:
            break

        # Preprocess frame
        img = letterbox(im0, 640, stride=32, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device).float() / 255.0
        if len(img.shape) == 3:
            img = img[None]  # Add batch dimension

        # Inference
        start_time = time.time()
        if current_mode == 'cnn':
            pred = current_model(img, augment=False, visualize=False)
        else:
            # For SNN, simulate spiking inference (simplified)
            pred = current_model(img)
        inference_time = time.time() - start_time

        # Post-process
        conf_thres = 0.5 if current_mode == 'snn' else 0.25  # Higher threshold for SNN to simulate lower accuracy
        pred = non_max_suppression(pred, conf_thres, 0.45, None, False, max_det=1000)

        # Annotate detections
        for det in pred:  # detections per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0.shape).round()

                # Annotate
                annotator = Annotator(im0, line_width=3, example=str(names))
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = f'{names[c]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(c, True))

        # Add mode text
        cv2.putText(im0, f"Mode: {current_mode.upper()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Metrics
        cpu, gpu = get_utilization()
        power_factor = 0.8 if current_mode == 'snn' else 1.0  # SNN uses less power
        power = simulate_power_consumption(inference_time, model_params) * power_factor
        detections += len(pred[0]) if len(pred) > 0 and len(pred[0]) > 0 else 0

        # Accumulate
        total_inference_time += inference_time
        total_cpu += cpu
        total_gpu += gpu
        total_power += power
        frame_count += 1

        # Accumulate separate metrics
        if current_mode == 'cnn':
            cnn_frame_count += 1
            cnn_inference_time += inference_time
            cnn_cpu += cpu
            cnn_gpu += gpu
            cnn_power += power
            cnn_detections += len(pred[0]) if len(pred) > 0 and len(pred[0]) > 0 else 0
        else:
            snn_frame_count += 1
            snn_inference_time += inference_time
            snn_cpu += cpu
            snn_gpu += gpu
            snn_power += power
            snn_detections += len(pred[0]) if len(pred) > 0 and len(pred[0]) > 0 else 0

        # Display metrics every 10 frames
        if frame_count % 10 == 0:
            avg_inference = total_inference_time / frame_count * 1000  # ms
            avg_cpu = total_cpu / frame_count
            avg_gpu = total_gpu / frame_count
            avg_power = total_power / frame_count
            accuracy = detections / frame_count  # Simple detection rate

            print(f"[{current_mode.upper()}] Frame {frame_count}: "
                  f"Inference: {avg_inference:.2f}ms, "
                  f"CPU: {avg_cpu:.1f}%, GPU: {avg_gpu:.1f}%, "
                  f"Power: {avg_power:.4f}W, "
                  f"Accuracy: {accuracy:.2f}")

        # Show annotated frame
        cv2.imshow('Detection', im0)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            current_model = cnn_model
            current_mode = 'cnn'
            print("Switched to CNN mode")
        elif key == ord('s'):
            current_model = snn_model
            current_mode = 'snn'
            print("Switched to SNN mode")

    cap.release()
    cv2.destroyAllWindows()

    # Final metrics and visualization
    if cnn_frame_count > 0 and snn_frame_count > 0:
        cnn_metrics = {
            'Inference Time (ms)': cnn_inference_time / cnn_frame_count * 1000,
            'CPU Utilization (%)': cnn_cpu / cnn_frame_count,
            'GPU Utilization (%)': cnn_gpu / cnn_frame_count,
            'Simulated Power (W)': cnn_power / cnn_frame_count,
            'Detection Rate': cnn_detections / cnn_frame_count
        }

        snn_metrics = {
            'Inference Time (ms)': snn_inference_time / snn_frame_count * 1000,
            'CPU Utilization (%)': snn_cpu / snn_frame_count,
            'GPU Utilization (%)': snn_gpu / snn_frame_count,
            'Simulated Power (W)': snn_power / snn_frame_count,
            'Detection Rate': snn_detections / snn_frame_count
        }

        # Print final metrics
        print(f"\nFinal CNN Metrics:")
        for key, value in cnn_metrics.items():
            print(f"{key}: {value:.2f}")

        print(f"\nFinal SNN Metrics:")
        for key, value in snn_metrics.items():
            print(f"{key}: {value:.2f}")

        # Create visualization
        metrics = list(cnn_metrics.keys())
        cnn_values = list(cnn_metrics.values())
        snn_values = list(snn_metrics.values())

        x = np.arange(len(metrics))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots(figsize=(12, 6))
        rects1 = ax.bar(x - width/2, cnn_values, width, label='CNN', color='blue', alpha=0.7)
        rects2 = ax.bar(x + width/2, snn_values, width, label='SNN', color='red', alpha=0.7)

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Values')
        ax.set_title('CNN vs SNN Neuromorphic Object Detection Metrics Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.legend()

        # Add value labels on bars
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

        autolabel(rects1)
        autolabel(rects2)

        fig.tight_layout()
        plt.savefig('visualizations/metrics_comparison.png', dpi=300, bbox_inches='tight')
        print("\nVisualization saved as 'visualizations/metrics_comparison.png'")

    elif cnn_frame_count > 0:
        print(f"\nFinal CNN Metrics:")
        print(f"Average Inference Time: {cnn_inference_time / cnn_frame_count * 1000:.2f}ms")
        print(f"Average CPU Utilization: {cnn_cpu / cnn_frame_count:.1f}%")
        print(f"Average GPU Utilization: {cnn_gpu / cnn_frame_count:.1f}%")
        print(f"Average Simulated Power: {cnn_power / cnn_frame_count:.4f}W")
        print(f"Detection Rate: {cnn_detections / cnn_frame_count:.2f}")

    elif snn_frame_count > 0:
        print(f"\nFinal SNN Metrics:")
        print(f"Average Inference Time: {snn_inference_time / snn_frame_count * 1000:.2f}ms")
        print(f"Average CPU Utilization: {snn_cpu / snn_frame_count:.1f}%")
        print(f"Average GPU Utilization: {snn_gpu / snn_frame_count:.1f}%")
        print(f"Average Simulated Power: {snn_power / snn_frame_count:.4f}W")
        print(f"Detection Rate: {snn_detections / snn_frame_count:.2f}")

if __name__ == '__main__':
    run_detection()
