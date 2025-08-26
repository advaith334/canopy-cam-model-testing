import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def process_images_in_folder_heavy(folder_path: str, resized_size: Tuple[int, int] = (320, 180)) -> None:
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        print(f"Invalid folder: {folder_path}")
        return

    # Output directory: <folder>/hsv_segmentation_heavy
    out_dir = folder / "hsv_segmentation_heavy"
    out_dir.mkdir(parents=True, exist_ok=True)

    # HSV green bounds (tunable)
    lower_green = (36, 20, 20)
    upper_green = (145, 255, 255)

    image_paths = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}
    ]

    # Pre-create structuring elements
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    for img_path in image_paths:
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"Failed to read image: {img_path}")
            continue

        # Resize to working resolution
        small_bgr = cv2.resize(bgr, resized_size, interpolation=cv2.INTER_LINEAR)

        # Denoise while preserving edges
        small_bgr = cv2.bilateralFilter(small_bgr, d=7, sigmaColor=75, sigmaSpace=75)

        # Convert to HSV
        hsv = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2HSV)

        # CLAHE on V channel to normalize lighting and enhance contrast
        h, s, v = cv2.split(hsv)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        v_eq = clahe.apply(v)
        hsv_eq = cv2.merge((h, s, v_eq))

        # Apply green threshold on equalized HSV
        mask = cv2.inRange(hsv_eq, lower_green, upper_green)

        # Suppress low-saturation areas to avoid false positives
        sat_thresh = 25
        sat_mask = cv2.threshold(s, sat_thresh, 255, cv2.THRESH_BINARY)[1]
        mask = cv2.bitwise_and(mask, sat_mask)

        # Morphological cleanup: open (remove noise) then close (fill gaps)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)

        # Remove small connected components (area filter)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels > 1:
            clean = np.zeros_like(mask)
            min_area = max(20, int(0.001 * (resized_size[0] * resized_size[1])))
            for label in range(1, num_labels):
                area = stats[label, cv2.CC_STAT_AREA]
                if area >= min_area:
                    clean[labels == label] = 255
            mask = clean

        # Percentage calculation at working scale
        green_pixels = int(cv2.countNonZero(mask))
        percentage = (green_pixels / float(resized_size[0] * resized_size[1])) * 100.0
        rounded = int(round(percentage))

        # Upscale mask to original resolution for output
        mask_upscaled = cv2.resize(mask, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Save as <stem>_<percentage>.jpeg in heavy output dir
        stem = img_path.stem
        out_path = out_dir / f"{stem}_{rounded}.jpeg"
        ok = cv2.imwrite(str(out_path), mask_upscaled)
        if not ok:
            print(f"Failed to write: {out_path}")


def main() -> int:
    if len(sys.argv) < 2:
        print(f"Usage: {Path(sys.argv[0]).name} <folder>")
        return 1
    process_images_in_folder_heavy(sys.argv[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
