import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def process_images_in_folder_kmeans(
    folder_path: str,
    resized_size: Tuple[int, int] = (320, 180),
    apply_clahe: bool = True,
) -> None:
    
    # K-means clustering (k=2) to separate plant vs non-plant pixels.

    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        print(f"Invalid folder: {folder_path}")
        return

    out_dir = folder / "k_means_clustering"
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

        # K-means clustering (k=2) on LAB features
        lab_small = cv2.merge((L, A, B)).astype(np.float32) / 255.0
        samples = lab_small.reshape((-1, 3))
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3)
        K = 2
        _, labels, centers = cv2.kmeans(
            samples,
            K,
            None,
            criteria,
            10,
            cv2.KMEANS_PP_CENTERS
        )

        # Determine which cluster represents plants using a composite heuristic
        # Compute per-cluster means in LAB and HSV spaces and spatial prior
        labels_flat = labels.flatten()
        h, w = resized_size[1], resized_size[0]

        # Prepare HSV for statistics
        hsv_for_stats = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2HSV)
        Hs, Ss, Vs = cv2.split(hsv_for_stats)

        # Build coordinate grid for spatial prior (favor lower 2/3 of image)
        yy = np.repeat(np.arange(h, dtype=np.int32)[:, None], w, axis=1)

        scores = []
        for c in range(K):
            mask_c = (labels_flat == c).reshape((h, w))
            count = int(np.count_nonzero(mask_c))
            if count == 0:
                scores.append(-1e9)
                continue

            # LAB stats (from normalized centers)
            a_mean = float(centers[c, 1])  # 0..1
            b_mean = float(centers[c, 2])  # 0..1

            # HSV stats (compute from pixels)
            H_mean = float(Hs[mask_c].mean()) if count > 0 else 0.0
            S_mean = float(Ss[mask_c].mean()) if count > 0 else 0.0
            V_mean = float(Vs[mask_c].mean()) if count > 0 else 0.0

            # Normalize HSV ranges to 0..1
            Hn = H_mean / 179.0
            Sn = S_mean / 255.0
            Vn = V_mean / 255.0

            # Hue closeness to green band (~35..85 on 0..179); compute proximity score
            green_low, green_high = 35.0/179.0, 85.0/179.0
            if Hn < green_low:
                hue_dist = green_low - Hn
            elif Hn > green_high:
                hue_dist = Hn - green_high
            else:
                hue_dist = 0.0
            hue_score = 1.0 - min(1.0, hue_dist / (50.0/179.0))  # within band => ~1

            # Spatial prior: fraction of pixels in lower 2/3 of the image
            lower_region = (yy >= (h // 3))
            spatial_frac = float(np.count_nonzero(mask_c & lower_region)) / float(count)

            # Composite score: lower A, higher S, good green hue, moderate V, more in lower region
            score = (
                (-a_mean) * 1.0 +
                (Sn) * 0.8 +
                (hue_score) * 1.0 +
                ((1.0 - Vn)) * 0.3 +
                (spatial_frac) * 0.5 +
                (b_mean) * 0.2
            )
            scores.append(score)

        plant_cluster = int(np.argmax(scores))

        # Create mask from plant cluster
        mask = (labels.flatten() == plant_cluster).astype(np.uint8) * 255
        mask = mask.reshape((resized_size[1], resized_size[0]))

        # Optional: clip by luminance to avoid sky/glare
        L_low, L_high = 20, 240
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

        # Calculate percentage at working scale
        plant_pixels = int(cv2.countNonZero(mask))
        percentage = (plant_pixels / float(resized_size[0] * resized_size[1])) * 100.0
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
    process_images_in_folder_kmeans(sys.argv[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
