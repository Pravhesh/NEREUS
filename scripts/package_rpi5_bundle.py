#!/usr/bin/env python3
"""
Package the self-contained deployment bundle for Raspberry Pi 5.
Assembles unquantized ONNX models, sample test images, synthetic generator,
and lightweight ARM64 requirements into deployment_rpi5/.
"""

import sys
import shutil
import argparse
from pathlib import Path

# Add project root and deployment folder to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from deployment_rpi5.test_images.generate_synthetic import generate_synthetic_image


REQUIREMENTS_RPI5_CONTENT = """# =============================================================================
# PlanktonAI Edge Runtime Dependencies - Raspberry Pi 5 (ARM64)
# =============================================================================
onnxruntime>=1.16.0
opencv-python-headless>=4.8.0
numpy<2.0.0
psutil>=5.9.0
"""


def package_bundle(
    models_dir: str = "models/onnx",
    bundle_dir: str = "deployment_rpi5",
    num_samples: int = 3,
) -> Path:
    """Assemble all assets into the deployment bundle directory."""
    bundle_p = Path(bundle_dir).resolve()
    models_src = Path(models_dir).resolve()

    models_dst = bundle_p / "models"
    test_img_dst = bundle_p / "test_images"

    models_dst.mkdir(parents=True, exist_ok=True)
    test_img_dst.mkdir(parents=True, exist_ok=True)

    print(f"\n[Packager] Creating deployment bundle at: {bundle_p}")

    # 1. Copy ONNX models
    expected_models = [
        "yolo26n_detector.onnx",
        "binary_filter.onnx",
        "mobilenet_classifier.onnx",
    ]
    for model_name in expected_models:
        src = models_src / model_name
        dst = models_dst / model_name
        if src.exists():
            print(f"  -> Copying {model_name} ({src.stat().st_size / (1024*1024):.2f} MB)")
            shutil.copy2(src, dst)
        else:
            print(f"  WARNING: {src} does not exist yet.")

    # 2. Provision test images
    packaged_count = 0

    # Look for existing test images in examples/ or runs/
    candidate_sources = [
        ROOT_DIR / "examples" / "output_visualization.jpg",
        ROOT_DIR / "runs" / "detect" / "val" / "val_batch0_labels.jpg",
        ROOT_DIR / "runs" / "detect" / "val" / "val_batch1_labels.jpg",
    ]
    for src in candidate_sources:
        if src.exists() and packaged_count < num_samples:
            packaged_count += 1
            dst = test_img_dst / f"sample_frame_{packaged_count:02d}.jpg"
            shutil.copy2(src, dst)
            print(f"  -> Included real sample: {dst.name}")

    # Fill remainder with synthetic images
    while packaged_count < num_samples:
        packaged_count += 1
        dst = test_img_dst / f"sample_frame_{packaged_count:02d}.jpg"
        generate_synthetic_image(str(dst), width=640, height=640, num_objects=8, seed=packaged_count * 31)
        print(f"  -> Generated synthetic sample: {dst.name}")

    # Copy generator script into test_images/ for on-device generation
    synth_src = ROOT_DIR / "deployment_rpi5" / "test_images" / "generate_synthetic.py"
    synth_dst = test_img_dst / "generate_synthetic.py"
    if synth_src.resolve() != synth_dst.resolve() and synth_src.exists():
        shutil.copy2(synth_src, synth_dst)

    # 3. Write requirements_rpi5.txt
    req_file = bundle_p / "requirements_rpi5.txt"
    req_file.write_text(REQUIREMENTS_RPI5_CONTENT)
    print("  -> Created requirements_rpi5.txt")

    print(f"[Packager] Deployment bundle successfully packaged at: {bundle_p}\n")
    return bundle_p


def main():
    parser = argparse.ArgumentParser(description="Package RPi5 ONNX deployment bundle")
    parser.add_argument("--models-dir", type=str, default="models/onnx", help="Directory with exported ONNX models")
    parser.add_argument("--bundle-dir", type=str, default="deployment_rpi5", help="Target deployment bundle directory")
    parser.add_argument("--num-samples", type=int, default=3, help="Number of test images to bundle")
    args = parser.parse_args()

    package_bundle(
        models_dir=args.models_dir,
        bundle_dir=args.bundle_dir,
        num_samples=args.num_samples,
    )


if __name__ == "__main__":
    main()
