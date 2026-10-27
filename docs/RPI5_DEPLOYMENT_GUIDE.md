# Raspberry Pi 5 Zooplankton Deployment & Benchmarking Guide

This guide walks through deploying the unquantized FP32 zooplankton detection, binary filtering, and taxonomic classification pipeline to a **Raspberry Pi 5** and accurately measuring its CPU performance, latency, and thermal behavior.

---

## 1. System Overview

The edge deployment architecture runs **unquantized FP32 ONNX models** using **ONNX Runtime with ARM XNNPACK multi-threading**:

```
+---------------------------------------------------------------------------------+
|                                 Raspberry Pi 5                                  |
|                                                                                 |
|   Input Frame (ZooScan / Microscope)                                            |
|        │                                                                        |
|        ▼                                                                        |
|   [Stage 1: Preprocess] ── Letterboxing & RGB normalization (416x416)           |
|        │                                                                        |
|        ▼                                                                        |
|   [Stage 2: YOLO26n Detector] ── ONNX Runtime (132 classes, conf>=0.25, iou>=0.45)|
|        │                                                                        |
|        ▼ (N detected bounding boxes)                                            |
|   [Stage 3: Binary Filter] ── Dynamic Batch [N, 3, 128, 128] (Plankton vs Artifact)|
|        │                                                                        |
|        ▼ (M surviving biological plankton)                                      |
|   [Stage 4: MobileNet Classifier] ── Dynamic Batch [M, 3, 224, 224] (120 Classes)  |
|                                     (Fused learned gamma preprocessor)          |
+---------------------------------------------------------------------------------+
```

### Why Unquantized ONNX on Pi 5?
* **Zero Accuracy Loss:** Keeps exact FP32 model weights from training.
* **Minimal Footprint:** No heavy PyTorch install required on the Pi SD card; runs on lightweight `onnxruntime` (~25 MB).
* **Multi-Core Scaling:** XNNPACK automatically distributes matrix operations across all 4 Cortex-A76 cores @ 2.4 GHz.

---

## 2. Hardware Preparation

* **Target Device:** Raspberry Pi 5 (4GB or 8GB RAM).
* **OS:** Raspberry Pi OS Bookworm (64-bit / `aarch64`) or Ubuntu 24.04 Server (64-bit).
* **Cooling (Crucial):** Use the official **Raspberry Pi Active Cooler** or an aluminum heatsink case with fan. Under continuous 4-core multi-threaded inference, uncooled boards will exceed 80 °C and thermally throttle, reducing clock speeds from 2.4 GHz to 1.5 GHz.

---

## 3. Step 1: Prepare the Deployment Bundle (Host PC)

Ensure your host PC has exported the latest ONNX models and packaged the bundle:

```bash
# 1. Export trained PyTorch weights to unquantized FP32 ONNX
python3 scripts/export_to_onnx.py --all --output-dir models/onnx

# 2. Package the self-contained bundle
python3 scripts/package_rpi5_bundle.py --bundle-dir deployment_rpi5 --num-samples 5
```

The resulting `deployment_rpi5/` folder contains:
* `models/`: `yolo26n_detector.onnx`, `binary_filter.onnx`, `mobilenet_classifier.onnx`
* `test_images/`: Packaged real test frames + `generate_synthetic.py`
* `benchmark_rpi5.py`: The benchmarking suite and telemetry logger
* `requirements_rpi5.txt`: Lightweight edge runtime dependencies
* `run_benchmark.sh`: One-click runner script

---

## 4. Step 2: Transfer Bundle to Raspberry Pi 5

Copy the `deployment_rpi5` directory to your Pi over the local network via SCP:

```bash
# From your PC:
scp -r deployment_rpi5 pi@<raspberry-pi-ip>:~/
```

*(Alternatively, copy the `deployment_rpi5` folder to a USB drive and plug it into the Pi).*

---

## 5. Step 3: Run the Benchmark on Raspberry Pi 5

SSH into the Raspberry Pi 5:

```bash
ssh pi@<raspberry-pi-ip>
cd ~/deployment_rpi5
```

Execute the automated benchmark runner:

```bash
./run_benchmark.sh
```

The script will automatically:
1. Create a lightweight local Python virtual environment (`venv_rpi5/`).
2. Install `onnxruntime`, `opencv-python-headless`, and `psutil`.
3. Check for test images (generates synthetic ones if none are found).
4. Run the benchmark across all 4 CPU cores and display the results.

---

## 6. Customizing Benchmark Options

You can pass arguments directly through `./run_benchmark.sh` or call `python3 benchmark_rpi5.py`:

```bash
# Run 100 iterations for thermal stress testing:
./run_benchmark.sh --iterations 100

# Benchmark using custom test images:
./run_benchmark.sh --images-dir /path/to/my_zooscan_images

# Test scaling with 2 threads instead of 4:
./run_benchmark.sh --threads 2

# Specify a custom JSON output path:
./run_benchmark.sh --out-report my_pi5_results.json
```

---

## 7. Understanding the Results

When the benchmark finishes, it outputs a formatted summary table:

```text
======================================================================
           RASPBERRY PI 5 PERFORMANCE BENCHMARK REPORT
======================================================================

[1] HARDWARE & SYSTEM STATUS:
  CPU Utilization   : 88.5%
  Memory Footprint  : 250.45 MB (Peak RSS)
  Core Temperature  : 52.4 °C
  Throttling Status : 0x0 (Optimal - No Throttling)

[2] ISOLATED MODEL MICRO-BENCHMARKS:
  +----------------------+------------+--------------+-----------+---------+
  | Model                | Batch Size | Mean Latency | P95 Lat.  | FPS     |
  +----------------------+------------+--------------+-----------+---------+
  | YOLO26n Detector     | 1          |    28.50 ms  |  30.10 ms |   35.1  |
  | Binary Filter        | 1          |     1.45 ms  |   1.60 ms |  689.6  |
  | Binary Filter        | 4          |     3.90 ms  |   4.10 ms | 1025.6  |
  | Binary Filter        | 8          |     6.80 ms  |   7.20 ms | 1176.5  |
  | MobileNet Classifier | 1          |     6.85 ms  |   7.20 ms |  145.9  |
  | MobileNet Classifier | 4          |    21.20 ms  |  22.50 ms |  188.7  |
  | MobileNet Classifier | 8          |    39.50 ms  |  41.00 ms |  202.5  |
  +----------------------+------------+--------------+-----------+---------+

[3] END-TO-END PIPELINE MACRO-BENCHMARKS:
  Total Pipeline Latency : 65.20 ms (P95: 78.4 ms)
  Full Pipeline FPS      : 15.33 FPS
  Avg Objects / Frame    : 14.0 detections (5.0 biological)

  Cascade Stage Breakdown:
    - Stage 1 Preprocess              :   2.40 ms  ( 3.7%)
    - Stage 2 Yolo                    :  32.10 ms  (49.2%)
    - Stage 3 Binary Filter           :   7.50 ms  (11.5%)
    - Stage 4 Classifier              :  23.20 ms  (35.6%)
======================================================================
```

### Key Metrics to Look For:
1. **Total Pipeline FPS:** Typically 15–25 FPS on Pi 5 CPU depending on organism density per frame.
2. **Throttling Status:** Should read `0x0 (Optimal - No Throttling)`. If `0x4` (Currently throttled) or `0x8` (Soft temperature limit) appears, active cooling is required.
3. **RAM Footprint:** Typically ~250 MB total RSS, leaving over 3.7 GB free on a 4GB Pi.
4. **Structured JSON:** Complete run data is recorded in `benchmark_report.json` for easy plotting and paper documentation.

---

## 8. Troubleshooting

* **Permission error on `vcgencmd`:**
  If temperature or throttling cannot be read, add the user to the `video` group:
  ```bash
  sudo usermod -aG video $USER
  ```
* **Virtual environment creation fails:**
  Ensure Python virtual environment package is installed:
  ```bash
  sudo apt install -y python3-venv python3-pip
  ```
