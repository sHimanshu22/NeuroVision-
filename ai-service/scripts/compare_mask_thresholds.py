
from pathlib import Path

import numpy as np
from PIL import Image

DATASET = Path(r"S:\NeuroVision\dataset\archive (4)\brisc2025")

MASKS = [
    DATASET / "segmentation_task/train/masks/brisc2025_train_01304_me_ax_t1.png",
    DATASET / "segmentation_task/test/masks/brisc2025_test_00352_me_ax_t1.png",
    DATASET / "segmentation_task/train/masks/brisc2025_train_04176_pi_co_t1.png",
    DATASET / "segmentation_task/test/masks/brisc2025_test_00825_pi_co_t1.png",
]

THRESHOLDS = [0, 1, 128, 254, 255]


def main():
    for path in MASKS:
        with Image.open(path) as image:
            mask = np.asarray(image.convert("L"))

        print(f"\n{path.name}")
        print(f"Shape: {mask.shape}")

        for threshold in THRESHOLDS:
            foreground = mask > threshold
            pixels = int(foreground.sum())
            percentage = 100 * pixels / foreground.size

            print(
                f"  mask > {threshold:3}: "
                f"{pixels:6} foreground pixels "
                f"({percentage:.4f}%)"
            )


if __name__ == "__main__":
    main()
