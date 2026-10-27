#!/usr/bin/env python3
"""
Export zooplankton detection, binary filter, and classification models to unquantized FP32 ONNX.
Supports dynamic batching on classification models and fuses learned contrast preprocessing.
"""

import sys
import os
import argparse
import shutil
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import torch
import onnx
from ultralytics import YOLO

from src.classification import TwoStageBinaryFilter, MobileNetClassifier


def export_yolo(weights_path: str, output_path: str, imgsz: int = 416) -> str:
    """Export YOLO model to standard unquantized FP32 ONNX."""
    print(f"\n[YOLO] Loading weights: {weights_path}")
    weights_p = Path(weights_path)
    if not weights_p.exists():
        raise FileNotFoundError(f"YOLO weights not found: {weights_path}")

    model = YOLO(str(weights_p))
    print(f"[YOLO] Exporting to ONNX at imgsz={imgsz} (FP32)...")
    exported_file = model.export(format="onnx", imgsz=imgsz, dynamic=False, half=False)
    
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    if Path(exported_file) != out_p:
        shutil.move(exported_file, str(out_p))

    # Verify model
    onnx_model = onnx.load(str(out_p))
    onnx.checker.check_model(onnx_model)
    print(f"[YOLO] Successfully exported and verified: {out_p}")
    return str(out_p)


def export_binary_filter(weights_path: str, output_path: str, input_size: int = 128) -> str:
    """Export TwoStageBinaryFilter to ONNX with dynamic batch axis."""
    print(f"\n[Binary Filter] Loading weights: {weights_path}")
    weights_p = Path(weights_path)
    if not weights_p.exists():
        raise FileNotFoundError(f"Binary filter weights not found: {weights_path}")

    filter_obj = TwoStageBinaryFilter(model_path=str(weights_p), input_size=input_size, device="cpu")
    model = filter_obj.model
    model.eval()

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(1, 3, input_size, input_size, dtype=torch.float32)

    print(f"[Binary Filter] Exporting with dynamic batching to: {out_p}")
    torch.onnx.export(
        model,
        dummy_input,
        str(out_p),
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )

    onnx_model = onnx.load(str(out_p), load_external_data=True)
    onnx.save(onnx_model, str(out_p), save_as_external_data=False)
    data_file = out_p.with_name(out_p.name + ".data")
    if data_file.exists():
        data_file.unlink()
    onnx.checker.check_model(onnx_model)
    print(f"[Binary Filter] Successfully exported and verified: {out_p}")
    return str(out_p)


def export_mobilenet(weights_path: str, output_path: str, num_classes: int = 120, input_size: int = 224) -> str:
    """Export MobileNet classifier with fused learned gamma preprocessor and dynamic batching."""
    print(f"\n[MobileNet] Loading weights: {weights_path}")
    weights_p = Path(weights_path)
    if not weights_p.exists():
        raise FileNotFoundError(f"MobileNet weights not found: {weights_path}")

    classifier = MobileNetClassifier(
        model_path=str(weights_p),
        num_classes=num_classes,
        input_size=input_size,
        device="cpu",
        embed_preprocessing=True,
    )
    full_model = classifier.full_model
    full_model.eval()

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.rand(1, 3, input_size, input_size, dtype=torch.float32)

    print(f"[MobileNet] Exporting fused model (preprocessing + backbone) to: {out_p}")
    torch.onnx.export(
        full_model,
        dummy_input,
        str(out_p),
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
    )

    onnx_model = onnx.load(str(out_p), load_external_data=True)
    onnx.save(onnx_model, str(out_p), save_as_external_data=False)
    data_file = out_p.with_name(out_p.name + ".data")
    if data_file.exists():
        data_file.unlink()
    onnx.checker.check_model(onnx_model)
    print(f"[MobileNet] Successfully exported and verified: {out_p}")
    return str(out_p)


def main():
    parser = argparse.ArgumentParser(description="Export zooplankton PyTorch models to unquantized FP32 ONNX")
    parser.add_argument("--all", action="store_true", help="Export all models (YOLO, binary filter, classifier)")
    parser.add_argument("--yolo", action="store_true", help="Export YOLO detector")
    parser.add_argument("--binary", action="store_true", help="Export Binary Filter")
    parser.add_argument("--classifier", action="store_true", help="Export MobileNet Classifier")
    parser.add_argument("--yolo-weights", type=str, default="yolo26n.pt", help="Path to YOLO weights (.pt)")
    parser.add_argument("--binary-weights", type=str, default="models/binary_filter_best.pt", help="Path to binary filter weights")
    parser.add_argument("--classifier-weights", type=str, default="models/mobilenetv3_classifier_best.pt", help="Path to classifier weights")
    parser.add_argument("--num-classes", type=int, default=120, help="Number of taxonomic classes")
    parser.add_argument("--output-dir", type=str, default="models/onnx", help="Directory to save exported ONNX models")

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prefer runs/ best.pt if exists for YOLO
    yolo_weights = args.yolo_weights
    yolo_best_run = Path("runs/detect/runs/detect/zooplankton_yolo26n/weights/best.pt")
    if yolo_best_run.exists() and args.yolo_weights == "yolo26n.pt":
        yolo_weights = str(yolo_best_run)

    if args.all or args.yolo:
        export_yolo(yolo_weights, str(out_dir / "yolo26n_detector.onnx"))

    if args.all or args.binary:
        export_binary_filter(args.binary_weights, str(out_dir / "binary_filter.onnx"))

    if args.all or args.classifier:
        export_mobilenet(args.classifier_weights, str(out_dir / "mobilenet_classifier.onnx"), num_classes=args.num_classes)

    print(f"\nAll requested exports complete! Saved to: {out_dir}")


if __name__ == "__main__":
    main()
