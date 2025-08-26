import sys
import math
from pathlib import Path
from typing import Tuple

import cv2


def process_images_in_folder(folder_path: str, resized_size: Tuple[int, int] = (320, 180)) -> None:

    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        print(f"Invalid folder: {folder_path}")
        return

    lower_green = (36, 20, 20)
    upper_green = (145, 255, 255)

    image_paths = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}
    ]

    for img_path in image_paths:
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"Failed to read image: {img_path}")
            continue

        small_bgr = cv2.resize(bgr, resized_size, interpolation=cv2.INTER_LINEAR)
        hsv = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_green, upper_green)

        green_pixels = int(cv2.countNonZero(mask))
        percentage = (green_pixels / float(resized_size[0] * resized_size[1])) * 100.0
        rounded = int(round(percentage))

        mask_upscaled = cv2.resize(mask, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

        stem = img_path.stem
        out_path = img_path.with_name(f"{stem}_{rounded}.jpeg")

        ok = cv2.imwrite(str(out_path), mask_upscaled)
        if not ok:
            print(f"Failed to write: {out_path}")


def main() -> int:
    if len(sys.argv) < 2:
        print(f"Usage: {Path(sys.argv[0]).name} <folder>")
        return 1
    process_images_in_folder(sys.argv[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


