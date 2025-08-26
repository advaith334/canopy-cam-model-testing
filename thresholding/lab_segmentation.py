import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def process_images_in_folder_lab(
    folder_path: str,
    resized_size: Tuple[int, int] = (320, 180),
    a_max: int = 135,
    b_min: int = 110,
    apply_clahe: bool = True,
) -> None:
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        print(f"Invalid folder: {folder_path}")
        return

    # Output directory: <folder>/lab_segmentation
    out_dir = folder / "lab_segmentation"
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}
    ]

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    for img_path in image_paths:
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"Failed to read image: {img_path}")
            continue

        # Resize and mild denoise
        small_bgr = cv2.resize(bgr, resized_size, interpolation=cv2.INTER_LINEAR)
        small_bgr = cv2.bilateralFilter(small_bgr, d=5, sigmaColor=60, sigmaSpace=60)

        # Convert to LAB
        lab = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)

        if apply_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            L = clahe.apply(L)
            lab = cv2.merge((L, A, B))
            # Re-split in case LUT changed shape
            L, A, B = cv2.split(lab)

        # Threshold: A low (<= a_max), B high (>= b_min)
        _, a_mask = cv2.threshold(A, a_max, 255, cv2.THRESH_BINARY_INV)
        _, b_mask = cv2.threshold(B, b_min, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_and(a_mask, b_mask)

        # Optional: limit to reasonable brightness to avoid glare/sky
        # Keep mid-to-high luminance but drop near-white saturation: 20..230
        L_low, L_high = 20, 230
        L_clip = cv2.inRange(L, L_low, L_high)
        mask = cv2.bitwise_and(mask, L_clip)

        # Morphological refine
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)

        # Remove tiny blobs
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels > 1:
            clean = np.zeros_like(mask)
            min_area = max(20, int(0.001 * (resized_size[0] * resized_size[1])))
            for label in range(1, num_labels):
                area = stats[label, cv2.CC_STAT_AREA]
                if area >= min_area:
                    clean[labels == label] = 255
            mask = clean

        # Percentage at working scale
        green_pixels = int(cv2.countNonZero(mask))
        percentage = (green_pixels / float(resized_size[0] * resized_size[1])) * 100.0
        rounded = int(round(percentage))

        # Save upscaled mask
        mask_upscaled = cv2.resize(mask, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
        out_path = out_dir / f"{img_path.stem}_{rounded}.jpeg"
        ok = cv2.imwrite(str(out_path), mask_upscaled)
        if not ok:
            print(f"Failed to write: {out_path}")


def main() -> int:
    if len(sys.argv) < 2:
        print(f"Usage: {Path(sys.argv[0]).name} <folder>")
        return 1
    process_images_in_folder_lab(sys.argv[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
