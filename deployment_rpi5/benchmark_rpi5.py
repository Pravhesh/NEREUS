#!/usr/bin/env python3
"""
Raspberry Pi 5 Unquantized ONNX Deployment Benchmark Harness.
Runs dual-mode performance evaluations:
1. Isolated Micro-benchmarks (pure model compute latency & throughput)
2. End-to-End Pipeline Macro-benchmarks (full cascade on macroscopic frames)
Collects comprehensive telemetry: CPU utilization, RAM (RSS), and thermal throttling.
"""

import os
import sys
import time
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import cv2
import numpy as np
import psutil
import onnxruntime as ort


# -----------------------------------------------------------------------------
# System Telemetry & Health Monitoring
# -----------------------------------------------------------------------------

class SystemTelemetry:
    """Collects system resource usage, thermals, and throttling flags on RPi 5."""

    def __init__(self):
        self.process = psutil.Process()
        self.has_vcgencmd = shutil_which("vcgencmd") is not None

    def get_temperature(self) -> Optional[float]:
        """Read CPU temperature in Celsius with multiple fallbacks."""
        # 1. Try vcgencmd (RPi native)
        if self.has_vcgencmd:
            try:
                out = subprocess.check_output(["vcgencmd", "measure_temp"], text=True)
                # output format: temp=48.2'C
                if "temp=" in out:
                    return float(out.split("=")[1].split("'")[0])
            except Exception:
                pass

        # 2. Try Linux thermal zone sysfs
        thermal_zone = Path("/sys/class/thermal/thermal_zone0/temp")
        if thermal_zone.exists():
            try:
                temp_raw = int(thermal_zone.read_text().strip())
                return temp_raw / 1000.0
            except Exception:
                pass

        return None

    def get_throttled_status(self) -> str:
        """Decode Raspberry Pi throttling bitmask."""
        if not self.has_vcgencmd:
            return "vcgencmd not available (non-RPi OS or permissions)"

        try:
            out = subprocess.check_output(["vcgencmd", "get_throttled"], text=True).strip()
            # format: throttled=0x0
            code_str = out.split("=")[1]
            code = int(code_str, 16)
            if code == 0:
                return "0x0 (Optimal - No Throttling)"

            issues = []
            if code & 0x1:
                issues.append("Under-voltage detected")
            if code & 0x2:
                issues.append("ARM frequency capped")
            if code & 0x4:
                issues.append("Currently throttled")
            if code & 0x8:
                issues.append("Soft temperature limit active")
            if code & 0x10000:
                issues.append("Under-voltage has occurred since boot")
            if code & 0x20000:
                issues.append("ARM frequency capping has occurred")
            if code & 0x40000:
                issues.append("Throttling has occurred")
            return f"{code_str}: " + ", ".join(issues)
        except Exception as e:
            return f"Error reading throttled status: {e}"

    def get_stats(self) -> Dict[str, Any]:
        """Collect snapshot of CPU, RAM, and thermals."""
        temp = self.get_temperature()
        mem_info = self.process.memory_info()
        return {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "ram_rss_mb": round(mem_info.rss / (1024 * 1024), 2),
            "ram_vms_mb": round(mem_info.vms / (1024 * 1024), 2),
            "temperature_c": round(temp, 1) if temp is not None else None,
            "throttled_status": self.get_throttled_status(),
        }


def shutil_which(cmd: str) -> Optional[str]:
    import shutil
    return shutil.which(cmd)


# -----------------------------------------------------------------------------
# ONNX Inference Pipeline (Full Cascade)
# -----------------------------------------------------------------------------

class ONNXPipeline:
    """Manages ONNX Runtime sessions and executes the multi-stage cascade."""

    def __init__(self, models_dir: str = "models", num_threads: int = 4):
        self.models_dir = Path(models_dir)
        self.num_threads = num_threads

        # Configure session options for ARM Cortex-A76
        self.session_opts = ort.SessionOptions()
        self.session_opts.intra_op_num_threads = num_threads
        self.session_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Load sessions
        yolo_path = self.models_dir / "yolo26n_detector.onnx"
        binary_path = self.models_dir / "binary_filter.onnx"
        cls_path = self.models_dir / "mobilenet_classifier.onnx"

        for p in [yolo_path, binary_path, cls_path]:
            if not p.exists():
                raise FileNotFoundError(f"Required model not found: {p}")

        print(f"[Pipeline] Initializing ONNX Runtime with {num_threads} CPU threads (XNNPACK)...")
        providers = ["CPUExecutionProvider"]
        self.yolo_sess = ort.InferenceSession(str(yolo_path), self.session_opts, providers=providers)
        self.binary_sess = ort.InferenceSession(str(binary_path), self.session_opts, providers=providers)
        self.cls_sess = ort.InferenceSession(str(cls_path), self.session_opts, providers=providers)

        self.yolo_input_name = self.yolo_sess.get_inputs()[0].name
        self.binary_input_name = self.binary_sess.get_inputs()[0].name
        self.cls_input_name = self.cls_sess.get_inputs()[0].name

        # Normalization constants for Binary Filter
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

    # ── Preprocessing & Letterboxing ─────────────────────────────────────────

    @staticmethod
    def letterbox(img: np.ndarray, new_shape: Tuple[int, int] = (416, 416), color: Tuple[int, int, int] = (114, 114, 114)) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Resize and pad image while meeting stride-multiple constraints."""
        shape = img.shape[:2]  # [height, width]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img, r, (left, top)

    # ── YOLO NMS ─────────────────────────────────────────────────────────────

    @staticmethod
    def postprocess_yolo(
        output: np.ndarray,
        orig_shape: Tuple[int, int],
        ratio: float,
        pad: Tuple[int, int],
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
    ) -> List[Dict[str, Any]]:
        """Parse YOLO raw output and run OpenCV NMS."""
        # Output is (1, 132, 3549) -> transpose to (3549, 132)
        preds = np.transpose(output[0], (1, 0))
        boxes = preds[:, :4]
        scores = preds[:, 4:]

        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)

        mask = confidences >= conf_thresh
        if not np.any(mask):
            return []

        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        # Convert center xywh to xyxy on 416x416
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = cx - w / 2 - pad[0]
        y1 = cy - h / 2 - pad[1]
        x2 = cx + w / 2 - pad[0]
        y2 = cy + h / 2 - pad[1]

        # Rescale to original image shape
        x1 = np.clip(x1 / ratio, 0, orig_shape[1])
        y1 = np.clip(y1 / ratio, 0, orig_shape[0])
        x2 = np.clip(x2 / ratio, 0, orig_shape[1])
        y2 = np.clip(y2 / ratio, 0, orig_shape[0])

        cv_boxes = [[int(x1[i]), int(y1[i]), int(x2[i] - x1[i]), int(y2[i] - y1[i])] for i in range(len(x1))]
        indices = cv2.dnn.NMSBoxes(cv_boxes, confidences.tolist(), conf_thresh, iou_thresh)

        detections = []
        if len(indices) > 0:
            for idx in indices.flatten():
                detections.append({
                    "bbox": (int(x1[idx]), int(y1[idx]), int(x2[idx]), int(y2[idx])),
                    "confidence": float(confidences[idx]),
                    "class_id": int(class_ids[idx]),
                })
        return detections

    # ── Full Frame Execution ─────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray, conf_thresh: float = 0.25, iou_thresh: float = 0.45) -> Dict[str, Any]:
        """Execute all pipeline stages and return precise timing breakdown."""
        timings = {}
        t_start = time.perf_counter_ns()

        # ── Stage 1: Preprocess & Letterbox ──────────────────────────────────
        t0 = time.perf_counter_ns()
        lb_img, ratio, pad = self.letterbox(frame, new_shape=(416, 416))
        # Convert BGR to RGB, HWC to CHW, normalize to [0, 1]
        yolo_input = lb_img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        yolo_input = np.expand_dims(yolo_input, axis=0)
        timings["stage1_preprocess_ms"] = (time.perf_counter_ns() - t0) / 1e6

        # ── Stage 2: YOLO Detection ──────────────────────────────────────────
        t1 = time.perf_counter_ns()
        yolo_raw = self.yolo_sess.run(None, {self.yolo_input_name: yolo_input})[0]
        detections = self.postprocess_yolo(yolo_raw, frame.shape[:2], ratio, pad, conf_thresh, iou_thresh)
        timings["stage2_yolo_ms"] = (time.perf_counter_ns() - t1) / 1e6

        num_det = len(detections)
        if num_det == 0:
            timings["stage3_binary_filter_ms"] = 0.0
            timings["stage4_classifier_ms"] = 0.0
            total_ms = (time.perf_counter_ns() - t_start) / 1e6
            return {
                "num_detections": 0,
                "num_biological": 0,
                "detections": [],
                "stage_times_ms": timings,
                "total_latency_ms": round(total_ms, 2),
            }

        # ── Stage 3: Crop Extraction & Binary Filter ─────────────────────────
        t2 = time.perf_counter_ns()
        crops = []
        valid_indices = []
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det["bbox"]
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
                continue
            crops.append(crop)
            valid_indices.append(i)

        if not crops:
            timings["stage3_binary_filter_ms"] = (time.perf_counter_ns() - t2) / 1e6
            timings["stage4_classifier_ms"] = 0.0
            total_ms = (time.perf_counter_ns() - t_start) / 1e6
            return {
                "num_detections": 0,
                "num_biological": 0,
                "detections": [],
                "stage_times_ms": timings,
                "total_latency_ms": round(total_ms, 2),
            }

        # Batch prepare for binary filter (128x128, ImageNet normalized)
        bf_batch = np.empty((len(crops), 3, 128, 128), dtype=np.float32)
        for idx, crop in enumerate(crops):
            resized = cv2.resize(crop, (128, 128))[:, :, ::-1] # BGR -> RGB
            bf_batch[idx] = (resized.transpose(2, 0, 1) / 255.0)

        bf_batch = (bf_batch - self.mean) / self.std
        bf_logits = self.binary_sess.run(None, {self.binary_input_name: bf_batch})[0]

        # Softmax on axis 1
        exp_logits = np.exp(bf_logits - np.max(bf_logits, axis=1, keepdims=True))
        bf_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        is_plankton = bf_probs[:, 1] > 0.5
        timings["stage3_binary_filter_ms"] = (time.perf_counter_ns() - t2) / 1e6

        # ── Stage 4: MobileNet Classifier on Biological Organisms ────────────
        t3 = time.perf_counter_ns()
        bio_indices = [i for i, plank in enumerate(is_plankton) if plank]

        class_predictions = {}
        if bio_indices:
            cls_batch = np.empty((len(bio_indices), 3, 224, 224), dtype=np.float32)
            for batch_pos, orig_idx in enumerate(bio_indices):
                crop = crops[orig_idx]
                resized = cv2.resize(crop, (224, 224))[:, :, ::-1] # BGR -> RGB
                # Preprocessor is fused into the graph: expects raw [0, 1] floats
                cls_batch[batch_pos] = (resized.transpose(2, 0, 1) / 255.0)

            cls_logits = self.cls_sess.run(None, {self.cls_input_name: cls_batch})[0]
            cls_preds = np.argmax(cls_logits, axis=1)

            for batch_pos, orig_idx in enumerate(bio_indices):
                class_predictions[orig_idx] = int(cls_preds[batch_pos])

        timings["stage4_classifier_ms"] = (time.perf_counter_ns() - t3) / 1e6

        # Assemble result
        final_detections = []
        for idx, orig_det_idx in enumerate(valid_indices):
            det = detections[orig_det_idx]
            is_bio = bool(is_plankton[idx])
            final_detections.append({
                "bbox": det["bbox"],
                "confidence": det["confidence"],
                "is_plankton": is_bio,
                "species_class": class_predictions.get(idx) if is_bio else None,
            })

        total_ms = (time.perf_counter_ns() - t_start) / 1e6
        return {
            "num_detections": len(final_detections),
            "num_biological": len(bio_indices),
            "detections": final_detections,
            "stage_times_ms": timings,
            "total_latency_ms": round(total_ms, 2),
        }


# -----------------------------------------------------------------------------
# Micro & Macro Benchmarking Functions
# -----------------------------------------------------------------------------

def run_micro_benchmarks(
    pipeline: ONNXPipeline,
    iterations: int = 50,
    batch_sizes: List[int] = [1, 4, 8],
) -> Dict[str, Any]:
    """Measure isolated model compute latency and throughput."""
    print(f"\n{'='*60}\nRUNNING ISOLATED MODEL MICRO-BENCHMARKS ({iterations} iterations)\n{'='*60}")
    results = {}

    # 1. YOLO26n Detector (fixed 416x416, batch=1)
    print("  -> Benchmarking YOLO26n Detector [1, 3, 416, 416]...")
    dummy_yolo = np.random.rand(1, 3, 416, 416).astype(np.float32)
    # Warmup
    for _ in range(5):
        pipeline.yolo_sess.run(None, {pipeline.yolo_input_name: dummy_yolo})

    times = []
    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        pipeline.yolo_sess.run(None, {pipeline.yolo_input_name: dummy_yolo})
        times.append((time.perf_counter_ns() - t0) / 1e6)

    mean_ms = float(np.mean(times))
    results["yolo"] = {
        1: {
            "mean_ms": round(mean_ms, 2),
            "p95_ms": round(float(np.percentile(times, 95)), 2),
            "std_ms": round(float(np.std(times)), 2),
            "fps": round(1000.0 / mean_ms, 2) if mean_ms > 0 else 0,
        }
    }

    # 2. Binary Filter across batch sizes
    results["binary_filter"] = {}
    for bs in batch_sizes:
        print(f"  -> Benchmarking Binary Filter [{bs}, 3, 128, 128]...")
        dummy_bf = np.random.rand(bs, 3, 128, 128).astype(np.float32)
        for _ in range(5):
            pipeline.binary_sess.run(None, {pipeline.binary_input_name: dummy_bf})

        times = []
        for _ in range(iterations):
            t0 = time.perf_counter_ns()
            pipeline.binary_sess.run(None, {pipeline.binary_input_name: dummy_bf})
            times.append((time.perf_counter_ns() - t0) / 1e6)

        mean_ms = float(np.mean(times))
        results["binary_filter"][bs] = {
            "mean_ms": round(mean_ms, 2),
            "p95_ms": round(float(np.percentile(times, 95)), 2),
            "std_ms": round(float(np.std(times)), 2),
            "fps": round((1000.0 / mean_ms) * bs, 2) if mean_ms > 0 else 0,
        }

    # 3. MobileNet Classifier across batch sizes
    results["classifier"] = {}
    for bs in batch_sizes:
        print(f"  -> Benchmarking MobileNet Classifier [{bs}, 3, 224, 224]...")
        dummy_cls = np.random.rand(bs, 3, 224, 224).astype(np.float32)
        for _ in range(5):
            pipeline.cls_sess.run(None, {pipeline.cls_input_name: dummy_cls})

        times = []
        for _ in range(iterations):
            t0 = time.perf_counter_ns()
            pipeline.cls_sess.run(None, {pipeline.cls_input_name: dummy_cls})
            times.append((time.perf_counter_ns() - t0) / 1e6)

        mean_ms = float(np.mean(times))
        results["classifier"][bs] = {
            "mean_ms": round(mean_ms, 2),
            "p95_ms": round(float(np.percentile(times, 95)), 2),
            "std_ms": round(float(np.std(times)), 2),
            "fps": round((1000.0 / mean_ms) * bs, 2) if mean_ms > 0 else 0,
        }

    return results


def run_macro_benchmarks(
    pipeline: ONNXPipeline,
    image_paths: List[Path],
    iterations: int = 50,
) -> Dict[str, Any]:
    """Run full end-to-end pipeline cascade on real test frames."""
    print(f"\n{'='*60}\nRUNNING END-TO-END PIPELINE MACRO-BENCHMARKS ({iterations} iterations)\n{'='*60}")
    frames = [cv2.imread(str(p)) for p in image_paths if cv2.imread(str(p)) is not None]
    if not frames:
        raise ValueError("No valid images loaded for macro benchmarking.")

    # Warmup
    for _ in range(3):
        pipeline.process_frame(frames[0])

    all_stage_times: Dict[str, List[float]] = {
        "stage1_preprocess_ms": [],
        "stage2_yolo_ms": [],
        "stage3_binary_filter_ms": [],
        "stage4_classifier_ms": [],
    }
    total_latencies = []
    total_detections = []
    total_biological = []

    print(f"  Executing across {len(frames)} test frames ({iterations} runs)...")
    for i in range(iterations):
        frame = frames[i % len(frames)]
        res = pipeline.process_frame(frame)
        total_latencies.append(res["total_latency_ms"])
        total_detections.append(res["num_detections"])
        total_biological.append(res["num_biological"])
        for stage, val in res["stage_times_ms"].items():
            all_stage_times[stage].append(val)

    mean_total = float(np.mean(total_latencies))
    stage_breakdown = {
        stage: round(float(np.mean(vals)), 2) for stage, vals in all_stage_times.items()
    }

    return {
        "mean_total_latency_ms": round(mean_total, 2),
        "p95_total_latency_ms": round(float(np.percentile(total_latencies, 95)), 2),
        "fps": round(1000.0 / mean_total, 2) if mean_total > 0 else 0,
        "avg_detections_per_frame": round(float(np.mean(total_detections)), 1),
        "avg_biological_per_frame": round(float(np.mean(total_biological)), 1),
        "stage_breakdown_ms": stage_breakdown,
    }


def print_summary_table(micro: Dict[str, Any], macro: Dict[str, Any], telemetry: Dict[str, Any]):
    """Print readable summary tables to terminal."""
    print("\n" + "=" * 70)
    print("           RASPBERRY PI 5 PERFORMANCE BENCHMARK REPORT")
    print("=" * 70)

    print("\n[1] HARDWARE & SYSTEM STATUS:")
    print(f"  CPU Utilization   : {telemetry.get('cpu_percent')}%")
    print(f"  Memory Footprint  : {telemetry.get('ram_rss_mb')} MB (Peak RSS)")
    temp_str = f"{telemetry.get('temperature_c')} °C" if telemetry.get('temperature_c') else "N/A"
    print(f"  Core Temperature  : {temp_str}")
    print(f"  Throttling Status : {telemetry.get('throttled_status')}")

    print("\n[2] ISOLATED MODEL MICRO-BENCHMARKS:")
    print("  +----------------------+------------+--------------+-----------+---------+")
    print("  | Model                | Batch Size | Mean Latency | P95 Lat.  | FPS     |")
    print("  +----------------------+------------+--------------+-----------+---------+")
    yolo_stat = micro["yolo"][1]
    print(f"  | YOLO26n Detector     | 1          | {yolo_stat['mean_ms']:>8.2f} ms | {yolo_stat['p95_ms']:>6.2f} ms | {yolo_stat['fps']:>6.1f}  |")

    for bs, stat in micro["binary_filter"].items():
        print(f"  | Binary Filter        | {bs:<10} | {stat['mean_ms']:>8.2f} ms | {stat['p95_ms']:>6.2f} ms | {stat['fps']:>6.1f}  |")

    for bs, stat in micro["classifier"].items():
        print(f"  | MobileNet Classifier | {bs:<10} | {stat['mean_ms']:>8.2f} ms | {stat['p95_ms']:>6.2f} ms | {stat['fps']:>6.1f}  |")
    print("  +----------------------+------------+--------------+-----------+---------+")

    print("\n[3] END-TO-END PIPELINE MACRO-BENCHMARKS:")
    print(f"  Total Pipeline Latency : {macro['mean_total_latency_ms']} ms (P95: {macro['p95_total_latency_ms']} ms)")
    print(f"  Full Pipeline FPS      : {macro['fps']} FPS")
    print(f"  Avg Objects / Frame    : {macro['avg_detections_per_frame']} detections ({macro['avg_biological_per_frame']} biological)")
    print("\n  Cascade Stage Breakdown:")
    total = macro["mean_total_latency_ms"]
    for stage, ms in macro["stage_breakdown_ms"].items():
        pct = (ms / total) * 100 if total > 0 else 0
        clean_name = stage.replace("stage", "Stage ").replace("_ms", "").replace("_", " ").title()
        print(f"    - {clean_name:<32}: {ms:>6.2f} ms  ({pct:>4.1f}%)")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Raspberry Pi 5 ONNX Performance Benchmark")
    parser.add_argument("--models-dir", type=str, default="models", help="Directory containing ONNX models")
    parser.add_argument("--images-dir", type=str, default="test_images", help="Directory containing test images")
    parser.add_argument("--iterations", type=int, default=30, help="Number of benchmark iterations")
    parser.add_argument("--threads", type=int, default=4, help="Number of CPU intra-op threads (default 4)")
    parser.add_argument("--out-report", type=str, default="benchmark_report.json", help="Path to save JSON report")
    args = parser.parse_args()

    telemetry = SystemTelemetry()
    pipeline = ONNXPipeline(models_dir=args.models_dir, num_threads=args.threads)

    # Resolve test images
    img_dir = Path(args.images_dir)
    images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    if not images:
        print(f"No images found in {img_dir}. Generating synthetic test frames...")
        from test_images.generate_synthetic import generate_synthetic_image
        img_dir.mkdir(parents=True, exist_ok=True)
        for i in range(3):
            p = img_dir / f"synthetic_frame_{i+1:02d}.jpg"
            generate_synthetic_image(str(p), width=640, height=640, num_objects=6, seed=i * 23)
            images.append(p)

    # Run Micro Benchmarks
    micro_results = run_micro_benchmarks(pipeline, iterations=args.iterations, batch_sizes=[1, 4, 8])

    # Run Macro Benchmarks
    macro_results = run_macro_benchmarks(pipeline, images, iterations=args.iterations)

    # Final Telemetry Snapshot
    final_telemetry = telemetry.get_stats()

    # Print Summary
    print_summary_table(micro_results, macro_results, final_telemetry)

    # Save JSON Report
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "num_threads": args.threads,
            "iterations": args.iterations,
            "models_dir": args.models_dir,
            "num_test_images": len(images),
        },
        "system_telemetry": final_telemetry,
        "micro_benchmarks": micro_results,
        "macro_benchmarks": macro_results,
    }
    with open(args.out_report, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Structured JSON benchmark report saved to: {args.out_report}")


if __name__ == "__main__":
    main()
