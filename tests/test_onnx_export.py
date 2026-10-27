import unittest
import os
import shutil
from pathlib import Path
import numpy as np
import torch
import onnx
import onnxruntime as ort

from scripts.export_to_onnx import (
    export_yolo,
    export_binary_filter,
    export_mobilenet,
)
from src.classification import TwoStageBinaryFilter, MobileNetClassifier


class TestONNXExport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = Path("tests/scratch_onnx")
        cls.test_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)

    def test_export_binary_filter_parity(self):
        """Binary filter ONNX export must have dynamic batch and match PyTorch within 1e-4."""
        weights_path = Path("models/binary_filter_best.pt")
        output_onnx = self.test_dir / "binary_filter.onnx"

        # Export
        export_binary_filter(str(weights_path), str(output_onnx), input_size=128)
        self.assertTrue(output_onnx.exists())

        # Validate ONNX model
        model = onnx.load(str(output_onnx))
        onnx.checker.check_model(model)

        # PyTorch inference
        pt_filter = TwoStageBinaryFilter(model_path=str(weights_path), input_size=128, device="cpu")
        pt_filter.model.eval()

        dummy_input = torch.randn(2, 3, 128, 128, dtype=torch.float32)
        with torch.no_grad():
            pt_out = pt_filter.model(dummy_input).numpy()

        # ONNX Runtime inference
        session = ort.InferenceSession(str(output_onnx), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        ort_out = session.run(None, {input_name: dummy_input.numpy()})[0]

        np.testing.assert_allclose(pt_out, ort_out, rtol=1e-3, atol=1e-4)

    def test_export_mobilenet_parity(self):
        """MobileNet ONNX export must fuse preprocessor, support dynamic batch, and match PyTorch."""
        weights_path = Path("models/mobilenetv3_classifier_best.pt")
        output_onnx = self.test_dir / "mobilenet_classifier.onnx"

        # Export
        export_mobilenet(str(weights_path), str(output_onnx), num_classes=120, input_size=224)
        self.assertTrue(output_onnx.exists())

        # Validate ONNX model
        model = onnx.load(str(output_onnx))
        onnx.checker.check_model(model)

        # PyTorch inference using full_model (fused preprocessor + backbone)
        pt_classifier = MobileNetClassifier(
            model_path=str(weights_path),
            num_classes=120,
            input_size=224,
            device="cpu",
            embed_preprocessing=True,
        )
        pt_full = pt_classifier.full_model
        pt_full.eval()

        dummy_input = torch.rand(2, 3, 224, 224, dtype=torch.float32)
        with torch.no_grad():
            pt_out = pt_full(dummy_input).numpy()

        # ONNX Runtime inference
        session = ort.InferenceSession(str(output_onnx), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        ort_out = session.run(None, {input_name: dummy_input.numpy()})[0]

        np.testing.assert_allclose(pt_out, ort_out, rtol=1e-3, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
