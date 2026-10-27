import unittest
from pathlib import Path
import numpy as np
import cv2

from deployment_rpi5.benchmark_rpi5 import (
    SystemTelemetry,
    ONNXPipeline,
    run_micro_benchmarks,
    run_macro_benchmarks,
)


class TestBenchmarkHarness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models_dir = Path("models/onnx")
        cls.test_images_dir = Path("deployment_rpi5/test_images")

    def test_system_telemetry(self):
        """Telemetry collector returns system stats and handles missing vcgencmd gracefully."""
        telemetry = SystemTelemetry()
        stats = telemetry.get_stats()
        self.assertIn("cpu_percent", stats)
        self.assertIn("ram_rss_mb", stats)
        self.assertIn("temperature_c", stats)
        self.assertGreaterEqual(stats["ram_rss_mb"], 0.0)

        throttled = telemetry.get_throttled_status()
        self.assertIsInstance(throttled, str)

    def test_onnx_pipeline_zero_detections_safety(self):
        """Pipeline must handle frames with zero detections gracefully without errors."""
        pipeline = ONNXPipeline(models_dir=str(self.models_dir), num_threads=2)

        # Blank black image has no plankton / zero detections
        blank_img = np.zeros((640, 640, 3), dtype=np.uint8)
        result = pipeline.process_frame(blank_img, conf_thresh=0.99) # high threshold ensures 0 detections

        self.assertEqual(result["num_detections"], 0)
        self.assertEqual(result["num_biological"], 0)
        self.assertIn("stage_times_ms", result)
        self.assertIn("total_latency_ms", result)

    def test_run_micro_benchmarks_quick(self):
        """Micro-benchmarking runs and produces structured timing dictionaries."""
        pipeline = ONNXPipeline(models_dir=str(self.models_dir), num_threads=2)
        results = run_micro_benchmarks(pipeline, iterations=2, batch_sizes=[1, 2])

        self.assertIn("yolo", results)
        self.assertIn("binary_filter", results)
        self.assertIn("classifier", results)

        for model_name in ["yolo", "binary_filter", "classifier"]:
            self.assertIn("mean_ms", results[model_name][1])
            self.assertIn("fps", results[model_name][1])

    def test_run_macro_benchmarks_quick(self):
        """Macro-benchmarking runs end-to-end pipeline and breaks down stages."""
        pipeline = ONNXPipeline(models_dir=str(self.models_dir), num_threads=2)
        images = list(self.test_images_dir.glob("*.jpg"))[:2]
        self.assertGreaterEqual(len(images), 1)

        macro_results = run_macro_benchmarks(pipeline, images, iterations=2)
        self.assertIn("mean_total_latency_ms", macro_results)
        self.assertIn("fps", macro_results)
        self.assertIn("stage_breakdown_ms", macro_results)


if __name__ == "__main__":
    unittest.main()
