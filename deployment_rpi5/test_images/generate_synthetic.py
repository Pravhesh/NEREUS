#!/usr/bin/env python3
"""
Synthetic zooplankton frame generator for offline benchmarking on edge devices.
Generates realistic dark-field microscopic frames containing simulated plankton
and artifact particles.
"""

import argparse
import random
from pathlib import Path
import cv2
import numpy as np


def generate_synthetic_image(
    output_path: str,
    width: int = 640,
    height: int = 640,
    num_objects: int = 8,
    seed: int = 42,
) -> str:
    """Generate a single realistic microscopic test image."""
    random.seed(seed)
    np.random.seed(seed)

    # Dark field background (average value ~25 with low Gaussian noise)
    base_val = np.random.randint(20, 35)
    img = np.full((height, width, 3), base_val, dtype=np.uint8)
    noise = np.random.normal(0, 3, (height, width, 3)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Add vignette / slight lighting gradient
    y_coords, x_coords = np.ogrid[:height, :width]
    center_y, center_x = height / 2, width / 2
    dist_from_center = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)
    max_dist = np.sqrt(center_x ** 2 + center_y ** 2)
    vignette = 1.0 - 0.15 * (dist_from_center / max_dist)
    img = np.clip(img * vignette[..., np.newaxis], 0, 255).astype(np.uint8)

    for i in range(num_objects):
        x = random.randint(60, width - 60)
        y = random.randint(60, height - 60)
        obj_type = random.choice(["copepod", "egg", "bubble", "fiber", "jelly"])
        brightness = random.randint(140, 240)

        if obj_type == "copepod":
            # Elongated body
            axes = (random.randint(20, 45), random.randint(10, 20))
            angle = random.randint(0, 180)
            cv2.ellipse(img, (x, y), axes, angle, 0, 360, (brightness, brightness, brightness), -1)
            # Add antenna line
            rad = np.deg2rad(angle)
            ant_end = (int(x + axes[0] * 1.5 * np.cos(rad)), int(y + axes[0] * 1.5 * np.sin(rad)))
            cv2.line(img, (x, y), ant_end, (brightness - 30, brightness - 30, brightness - 30), 2)

        elif obj_type == "egg":
            # Smooth circle
            radius = random.randint(12, 28)
            cv2.circle(img, (x, y), radius, (brightness, brightness, brightness), -1)
            # Inner nucleus
            cv2.circle(img, (x, y), max(4, radius // 3), (brightness - 50, brightness - 50, brightness - 50), -1)

        elif obj_type == "bubble":
            # Bright ring with dark center
            radius = random.randint(15, 30)
            cv2.circle(img, (x, y), radius, (255, 255, 255), 2)
            cv2.circle(img, (x, y), radius - 3, (15, 15, 15), -1)

        elif obj_type == "fiber":
            # Thin curved white artifact line
            pts = np.array([
                [x, y],
                [x + random.randint(-30, 30), y + random.randint(20, 50)],
                [x + random.randint(-50, 50), y + random.randint(60, 100)]
            ], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(img, [pts], isClosed=False, color=(240, 240, 240), thickness=2)

        elif obj_type == "jelly":
            # Diffuse semi-transparent blob
            radius = random.randint(25, 55)
            overlay = img.copy()
            cv2.circle(overlay, (x, y), radius, (brightness - 40, brightness - 40, brightness - 40), -1)
            cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    # Slight Gaussian blur to mimic microscope focal plane
    img = cv2.GaussianBlur(img, (3, 3), 0)

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_p), img)
    return str(out_p)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic zooplankton microscope test frames")
    parser.add_argument("--count", type=int, default=5, help="Number of frames to generate")
    parser.add_argument("--out-dir", type=str, default=".", help="Directory to save images")
    parser.add_argument("--width", type=int, default=640, help="Image width")
    parser.add_argument("--height", type=int, default=640, help="Image height")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.count):
        fname = out_dir / f"synthetic_frame_{i+1:02d}.jpg"
        generate_synthetic_image(str(fname), width=args.width, height=args.height, num_objects=random.randint(4, 12), seed=i * 17 + 42)
        print(f"Generated: {fname}")


if __name__ == "__main__":
    main()
