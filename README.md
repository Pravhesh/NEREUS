# NEREUS: Automated Edge Classification & Enumeration of Marine Zooplankton

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Platform: Linux / Raspberry Pi 5](https://img.shields.io/badge/Platform-RPi%205%20%7C%20Linux%20x86__64-brightgreen.svg)](#hardware-deployment)
[![Inference: ONNX Runtime FP32](https://img.shields.io/badge/Inference-ONNX%20Runtime%20FP32-orange.svg)](#edge-performance-benchmarks)
[![Models: Hugging Face](https://img.shields.io/badge/Hugging%20Face-Models%20Repo-yellow.svg)](https://huggingface.co/Pravhesh/zooplankton_classifier)

NEREUS (**Next-generation Embedded Recognition & Enumeration of Underwater Species**) is a high-throughput, multi-stage computer vision pipeline engineered to detect, filter, and classify marine zooplankton from microscopic and ZooScan imaging systems.

Designed for field oceanography and autonomous edge deployments, NEREUS achieves **15–24 FPS** on a standard quad-core **Raspberry Pi 5** CPU under full **unquantized FP32 precision**, eliminating cloud connectivity requirements and preserving taxonomic identification accuracy across 120 marine categories.

---

## Architecture Overview

High-resolution underwater imaging captures wide macroscopic fields containing mixed biological specimens, debris, and optical artifacts. NEREUS resolves this through a four-stage hierarchical cascade:

```
[Input Frame] (ZooScan / Digital Microscope)
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 0: Classical Preprocessing & Adaptive Contrast            │
│ • Local Contrast Normalization (LCN) & CLAHE                    │
│ • Optical artifact rejection (bubbles, fibers, sensor scratches) │
│ • Optional marker-controlled watershed for clumped specimens    │
└─────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 1: Primary Localization (YOLO26n Detector)                │
│ • Input: 416 × 416 px, FP32                                     │
│ • Ultralight anchor-free detector (2.47M params, 2.7 GFLOPs)    │
│ • Vectorized OpenCV NMS (IoU ≥ 0.45, Conf ≥ 0.25)               │
└─────────────────────────────────────────────────────────────────┘
      │  Produces N candidate bounding boxes
      ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 2: Binary Gatekeeper Filter (MobileNetV3-Small)            │
│ • Input: [N, 3, 128, 128] dynamic batch, ImageNet normalized    │
│ • Rapidly separates true plankton from detritus/bubbles         │
│ • 94.0% binary validation accuracy, 0.66 ms latency             │
└─────────────────────────────────────────────────────────────────┘
      │  Passes M biological organisms (discards artifacts)
      ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 3: Fine-Grained Taxonomic Classifier (MobileNetV4-Small)  │
│ • Input: [M, 3, 224, 224] dynamic batch                         │
│ • Fused learned contrast preprocessor (γ = 0.7162)              │
│ • 120-class classification across 980k dataset images           │
│ • 87.1% top-1 validation accuracy, 2.82 ms latency              │
└─────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 4: Ecological Analytics & Structured Export               │
│ • Taxonomic census & biomass estimation                         │
│ • Shannon-Wiener Diversity (H'), Simpson (D), Evenness (J')     │
│ • Standardized Darwin Core (DwC) & CSV/JSON export              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Model Specifications & Benchmarks

All models were trained on Amrita HPC using NVIDIA Tesla V100 GPUs on a curated dataset of **980,398 images** across **120 classes** (807,423 training, 172,975 validation) derived from the Bay of Biscay PELGAS survey archive.

### 1. Model Summary

| Stage | Model Architecture | Parameters | Input Resolution | Task Accuracy / Metric | Inference Latency (V100 GPU) | Inference Latency (RPi 5 CPU) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Stage 1** | YOLO26n Detector | 2.47M | 416 × 416 | 68.41% mAP@0.5 (173k instances) | 0.9 ms | 13.79 ms |
| **Stage 2** | Two-Stage Binary Filter | 0.93M | 128 × 128 | 94.0% Val Accuracy (loss 0.144) | 0.2 ms | 0.66 ms |
| **Stage 3** | MobileNetV4-Conv-Small | 2.65M | 224 × 224 | 87.1% Top-1 Val Accuracy | 0.8 ms | 2.82 ms |
| **Stage 3 (Pruned)** | MobileNetV4 Structured | 2.08M | 224 × 224 | 85.3% Val Accuracy (21.6% sparse) | 0.6 ms | 2.21 ms |

### 2. Edge Hardware Benchmarks (Raspberry Pi 5)

Evaluated on an unthrottled **Raspberry Pi 5 (ARM Cortex-A76 quad-core @ 2.4 GHz, 8GB RAM)** using `onnxruntime` with ARM XNNPACK multithreading (4 threads) under full FP32 precision:

| Benchmark Mode | Workload / Metric | Mean Latency | 95th Percentile | Throughput |
| :--- | :--- | :--- | :--- | :--- |
| **Micro-Benchmark** | YOLO26n (Batch = 1) | 13.79 ms | 13.81 ms | 72.5 FPS |
| **Micro-Benchmark** | Binary Filter (Batch = 1) | 0.66 ms | 0.68 ms | 1,513.3 FPS |
| **Micro-Benchmark** | Binary Filter (Batch = 4) | 1.59 ms | 1.61 ms | 2,513.8 FPS |
| **Micro-Benchmark** | Binary Filter (Batch = 8) | 2.67 ms | 2.68 ms | 2,996.6 FPS |
| **Micro-Benchmark** | MobileNetV4 (Batch = 1) | 2.82 ms | 3.06 ms | 355.2 FPS |
| **Micro-Benchmark** | MobileNetV4 (Batch = 4) | 9.43 ms | 10.06 ms | 424.0 FPS |
| **Micro-Benchmark** | MobileNetV4 (Batch = 8) | 19.86 ms | 20.36 ms | 402.7 FPS |
| **Macro-Benchmark** | **End-to-End Pipeline** | **41.47 ms** | **56.70 ms** | **24.11 FPS** |

* **Memory Footprint:** 250.45 MB peak RSS (leaves >3.7 GB free on 4GB Pi 5).
* **Thermal Envelope:** 52.4 °C with Raspberry Pi Active Cooler (throttling status `0x0`).

---

## Repository Structure

```text
sih2026/
├── deployment_rpi5/               # Self-contained Raspberry Pi 5 deployment package
│   ├── benchmark_rpi5.py         # Dual-mode benchmark harness & telemetry logger
│   ├── requirements_rpi5.txt     # Lightweight ARM64 dependencies (Zero PyTorch)
│   ├── run_benchmark.sh          # One-click automated execution runner
│   ├── models/                   # Unquantized FP32 ONNX model files
│   └── test_images/              # Test sample frames & synthetic frame generator
├── docs/
│   ├── README_HPC.md             # PBS cluster job submission and training guide
│   └── RPI5_DEPLOYMENT_GUIDE.md  # Raspberry Pi 5 setup, execution, and tuning guide
├── hpc/                          # PBS cluster scripts for Amrita HPC training
│   ├── submit_job.pbs            # 96-hour master training job (YOLO + Classifier)
│   ├── submit_job_2.pbs          # Standalone post-training structured pruning job
│   └── upload_yolo.pbs           # Model artifact synchronization to Hugging Face
├── models/                       # Trained PyTorch checkpoints (.pt)
│   ├── binary_filter_best.pt     # Best binary gatekeeper weights (4.0 MB)
│   ├── mobilenetv3_classifier_best.pt # Best 120-class classifier weights (10.8 MB)
│   └── onnx/                     # Exported FP32 ONNX graphs
├── runs/                         # Training metrics, PR curves, confusion matrices
├── scripts/
│   ├── export_to_onnx.py         # PyTorch -> FP32 ONNX exporter with dynamic batching
│   ├── package_rpi5_bundle.py    # Deployment bundle packager & test provisioner
│   ├── prepare_dataset.py        # Dataset preprocessing, split, and YOLO formatting
│   ├── prune_only.py             # Post-training structured pruning & fine-tuning
│   └── train_models.py           # Multi-task training script with auto-class detection
├── src/                          # Canonical Python library
│   ├── classification/           # MobileNetV4 classifier & ZooScanPreprocessor
│   ├── detection/                # YOLO26n detector wrapper
│   ├── pipeline/                 # End-to-end ZooplanktonPipeline coordinator
│   └── preprocessing/            # CLAHE, artifact removal, watershed segmentation
├── tests/                        # Automated unit & integration tests
│   ├── test_benchmark_harness.py # Benchmark suite and zero-detection safety tests
│   ├── test_onnx_export.py       # Numerical parity tests (PyTorch vs ONNX Runtime)
│   └── test_package_bundle.py    # Deployment packager integrity tests
├── config.example.py             # Configuration credential template
└── requirements.txt              # Host Python development dependencies
```

---

## Quickstart: Python API

### Installation

For development and model export on a workstation:

```bash
# Clone the repository
git clone https://github.com/Pravhesh/NEREUS.git
cd NEREUS

# Setup environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Full Pipeline Inference

```python
import cv2
from src.pipeline import ZooplanktonPipeline

# Initialize pipeline (loads PyTorch weights)
pipeline = ZooplanktonPipeline(
    detector_path="runs/detect/runs/detect/zooplankton_yolo26n/weights/best.pt",
    classifier_path="models/mobilenetv3_classifier_best.pt",
    num_classes=120,
    device="cpu", # or "cuda"
    enable_preprocessing=True
)

# Process frame
frame = cv2.imread("examples/output_visualization.jpg")
results = pipeline.process_image(frame, return_visualization=True)

print(f"Detections: {results['num_detections']}")
for det in results['detections']:
    print(f"Class: {det['top_species']} | Conf: {det['confidence']:.2f} | BBox: {det['bbox']}")

# Calculate ecological metrics
species_counts = pipeline.count_species(frame)
diversity = pipeline.calculate_diversity_indices(species_counts)
print(f"Shannon-Wiener Index (H'): {diversity['shannon_index']:.3f}")
print(f"Simpson Dominance (D):     {diversity['simpson_index']:.3f}")
```

---

## Edge Deployment on Raspberry Pi 5

The `deployment_rpi5/` directory is an isolated, self-contained bundle that operates with **zero PyTorch dependencies** using `onnxruntime` optimized for ARM Cortex-A76.

### 1. Export Models to ONNX (Host PC)

Export the PyTorch checkpoints into single-file FP32 ONNX graphs with dynamic batching:

```bash
python3 scripts/export_to_onnx.py --all --output-dir models/onnx
python3 scripts/package_rpi5_bundle.py --bundle-dir deployment_rpi5 --num-samples 5
```

### 2. Transfer and Execute on Raspberry Pi 5

Copy the directory to the Raspberry Pi over the local network:

```bash
# From your PC
scp -r deployment_rpi5 pi@<rpi5-ip>:~/
```

SSH into the Raspberry Pi 5 and execute the one-click runner:

```bash
ssh pi@<rpi5-ip>
cd ~/deployment_rpi5
chmod +x run_benchmark.sh
./run_benchmark.sh
```

The script will automatically configure a virtual environment, install ARM64 wheels, run 50 benchmark iterations, monitor system thermals, and output a structured `benchmark_report.json`.

For detailed hardware configuration and tuning options, consult [`docs/RPI5_DEPLOYMENT_GUIDE.md`](docs/RPI5_DEPLOYMENT_GUIDE.md).

---

## Training & HPC Workflow

Training on large-scale datasets (such as ZooScan Bay of Biscay) is orchestrated via PBS scripts configured for high-performance clusters.

### 1. Dataset Preparation

```bash
python3 scripts/prepare_dataset.py \
    --input-dir individual_images \
    --output-dir prepared_dataset \
    --val-split 0.15 \
    --test-split 0.05
```

### 2. Cluster Training via PBS

Submit the training job to the cluster scheduler:

```bash
qsub hpc/submit_job.pbs
```

The script automatically allocates 8 CPU cores, 16 GB RAM, and an NVIDIA V100 GPU. It runs 100 epochs of YOLO26n training, trains the binary filter, and conducts 100 epochs of MobileNetV4 training with adaptive gamma learning.

### 3. Structured Pruning

To compress the classifier by ~21% without retraining from scratch:

```bash
python3 scripts/prune_only.py \
    --classification-dir prepared_dataset/classification \
    --weights models/mobilenetv3_classifier_best.pt \
    --prune-ratio 0.30 \
    --prune-structured \
    --prune-epochs 10 \
    --device cuda
```

Consult [`docs/README_HPC.md`](docs/README_HPC.md) for full cluster configuration parameters.

---

## Ecological Analytics Formulation

NEREUS calculates standardized biodiversity metrics directly from the detection-classification census:

1. **Shannon-Wiener Diversity Index ($H'$):**
   $$H' = -\sum_{i=1}^{S} p_i \ln p_i$$
   where $p_i = \frac{n_i}{N}$ is the proportion of total organisms belonging to species $i$.

2. **Simpson's Dominance Index ($D$):**
   $$D = \sum_{i=1}^{S} p_i^2$$

3. **Pielou's Evenness Index ($J'$):**
   $$J' = \frac{H'}{\ln S}$$
   quantifying equitability across species distributions independent of sample size.

Data can be exported directly into **Darwin Core Archive (DwC-A)** formatted TSV/CSV tables for direct ingestion into OBIS (Ocean Biodiversity Information System) and GBIF repositories.

---

## Verification & Testing

The repository maintains an automated unit and integration test suite:

```bash
python3 -m unittest discover tests -v
```

Tests cover:
* `test_onnx_export.py`: Parity verification between PyTorch and ONNX Runtime ($|y_{\text{pt}} - y_{\text{onnx}}| < 10^{-4}$).
* `test_package_bundle.py`: Verification of the self-contained deployment bundle and test generators.
* `test_benchmark_harness.py`: Pipeline safety under zero-detection edge cases and thermal telemetry parsers.

---

## Pretrained Weights & Artifacts

Pretrained model weights are hosted on the Hugging Face Hub:

* **Repository:** [`Pravhesh/zooplankton_classifier`](https://huggingface.co/Pravhesh/zooplankton_classifier)
* `yolo26n_best.pt`: YOLO26n detector weights (5.7 MB)
* `binary_filter_best.pt`: Two-stage binary artifact filter (4.1 MB)
* `mobilenetv3_classifier_best.pt`: 120-class taxonomic classifier (10.8 MB)

---

## Citation & License

This project is licensed under the [MIT License](LICENSE).

```bibtex
@software{nereus2026,
  author = {Pravhesh V},
  title = {NEREUS: Automated Edge Classification and Enumeration of Marine Zooplankton},
  year = {2026},
  url = {https://github.com/Pravhesh/NEREUS}
}
```
