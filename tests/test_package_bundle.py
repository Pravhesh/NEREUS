import unittest
import shutil
from pathlib import Path
import cv2
import numpy as np

from scripts.package_rpi5_bundle import (
    package_bundle,
    generate_synthetic_image,
)


class TestPackageBundle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_target = Path("tests/scratch_bundle")
        if cls.test_target.exists():
            shutil.rmtree(cls.test_target)

    @classmethod
    def tearDownClass(cls):
        if cls.test_target.exists():
            shutil.rmtree(cls.test_target)

    def test_generate_synthetic_image(self):
        """Synthetic frame generator creates a readable image with features."""
        out_path = self.test_target / "test_synth.jpg"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        generate_synthetic_image(str(out_path), width=640, height=640, num_objects=5)
        self.assertTrue(out_path.exists())
        img = cv2.imread(str(out_path))
        self.assertIsNotNone(img)
        self.assertEqual(img.shape, (640, 640, 3))
        # Ensure image is not blank
        self.assertGreater(float(np.std(img)), 5.0)

    def test_package_bundle_creates_structure(self):
        """Packager creates complete deployment directory structure."""
        bundle_dir = self.test_target / "deployment_rpi5"
        models_dir = Path("models/onnx")
        
        package_bundle(
            models_dir=str(models_dir),
            bundle_dir=str(bundle_dir),
            num_samples=3,
        )

        # Check models
        self.assertTrue((bundle_dir / "models" / "yolo26n_detector.onnx").exists())
        self.assertTrue((bundle_dir / "models" / "binary_filter.onnx").exists())
        self.assertTrue((bundle_dir / "models" / "mobilenet_classifier.onnx").exists())

        # Check test_images
        test_images = list((bundle_dir / "test_images").glob("*.jpg"))
        self.assertGreaterEqual(len(test_images), 3)
        self.assertTrue((bundle_dir / "test_images" / "generate_synthetic.py").exists())

        # Check requirements_rpi5.txt
        req_file = bundle_dir / "requirements_rpi5.txt"
        self.assertTrue(req_file.exists())
        content = req_file.read_text()
        self.assertIn("onnxruntime", content)
        self.assertIn("opencv-python-headless", content)
        self.assertIn("psutil", content)
        self.assertIn("numpy<2", content)


if __name__ == "__main__":
    unittest.main()
