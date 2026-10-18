
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image

DATASET = Path(r"S:\NeuroVision\dataset\archive (4)\brisc2025")

MASKS = [
    DATASET / "segmentation_task/train/masks/brisc2025_train_01304_me_ax_t1.png",
    DATASET / "segmentation_task/test/masks/brisc2025_test_00352_me_ax_t1.png",
    DATASET / "segmentation_task/train/masks/brisc2025_train_04176_pi_co_t1.png",
    DATASET / "segmentation_task/test/masks/brisc2025_test_00825_pi_co_t1.png",
]

for path in MASKS:
    with Image.open(path) as image:
        mask = np.asarray(image)

    values, counts = np.unique(mask, return_counts=True)
    frequencies = sorted(
        zip(values.tolist(), counts.tolist()),
        key=lambda item: item[1],
        reverse=True,
    )

    print(f"\nFile: {path.name}")
    print(f"Shape: {mask.shape}")
    print(f"Dtype: {mask.dtype}")
    print(f"Unique grayscale values: {len(values)}")
    print(f"Minimum / maximum: {mask.min()} / {mask.max()}")
    print("Most frequent values (value: pixel count):")
    for value, count in frequencies[:15]:
        print(f"  {value}: {count}")
