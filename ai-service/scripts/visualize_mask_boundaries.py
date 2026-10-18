
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "segmentation_manifest.csv"
OUTPUT = ROOT / "data" / "mask_boundary_inspection.png"

IMAGE_SIZE = 256

# Representative examples identified during our previous audits.
TARGETS = [
    "brisc2025_train_01304_me_ax_t1.jpg",
    "brisc2025_test_00352_me_ax_t1.jpg",
    "brisc2025_train_04176_pi_co_t1.jpg",
    "brisc2025_test_00825_pi_co_t1.jpg",
]


def main():
    rows = {}

    with MANIFEST.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        for row in csv.DictReader(file):
            if row["filename"] in TARGETS:
                rows[row["filename"]] = row

    missing = [name for name in TARGETS if name not in rows]
    if missing:
        raise RuntimeError(f"Files not found in manifest: {missing}")

    fig, axes = plt.subplots(
        len(TARGETS), 4, figsize=(16, 4 * len(TARGETS))
    )

    for i, filename in enumerate(TARGETS):
        row = rows[filename]

        with Image.open(row["image_path"]) as source:
            mri = source.convert("L").resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                Image.Resampling.BILINEAR,
            )
            mri = np.asarray(mri, dtype=np.float32) / 255.0

        with Image.open(row["mask_path"]) as source:
            original = np.asarray(source.convert("L"), dtype=np.uint8)

        resized_image = Image.fromarray(original).resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.NEAREST,
        )
        mask = np.asarray(resized_image, dtype=np.uint8)

        binary_128 = mask >= 128
        binary_255 = mask == 255
        intermediate = (mask > 0) & (mask < 255)

        axes[i, 0].imshow(original, cmap="gray", vmin=0, vmax=255)
        axes[i, 0].set_title(
            f"{filename}\nOriginal mask ({original.shape[1]}x{original.shape[0]})"
        )

        axes[i, 1].imshow(mask, cmap="gray", vmin=0, vmax=255)
        axes[i, 1].set_title(
            f"Nearest-neighbor resized mask\n"
            f"Unique values: {np.unique(mask).size}"
        )

        axes[i, 2].imshow(binary_128, cmap="gray", vmin=0, vmax=1)
        axes[i, 2].set_title(
            f"mask >= 128\nForeground: {binary_128.sum()} pixels"
        )

        axes[i, 3].imshow(mri, cmap="gray", vmin=0, vmax=1)
        axes[i, 3].imshow(
            np.ma.masked_where(~binary_128, binary_128),
            cmap="autumn",
            alpha=0.55,
            vmin=0,
            vmax=1,
        )
        axes[i, 3].set_title(
            f"MRI overlay\n"
            f"mask == 255: {binary_255.sum()} px\n"
            f"Intermediate: {intermediate.sum()} px"
        )

        for ax in axes[i]:
            ax.axis("off")

    fig.suptitle(
        "NeuroVision — Original Mask and Threshold Inspection",
        fontsize=15,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=130, bbox_inches="tight")
    plt.close(fig)

    print("Boundary inspection completed.")
    print("Preview:", OUTPUT)


if __name__ == "__main__":
    main()
