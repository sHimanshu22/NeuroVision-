
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


DATASET = Path(r"S:\NeuroVision\dataset\archive (4)\brisc2025")
OUTPUT = Path(__file__).resolve().parents[1] / "data" / "duplicate_mask_visuals"

PAIRS = [
    (
        "brisc2025_train_04194_pi_co_t1",
        "brisc2025_test_00827_pi_co_t1",
    ),
    (
        "brisc2025_train_04176_pi_co_t1",
        "brisc2025_test_00825_pi_co_t1",
    ),
    (
        "brisc2025_train_01304_me_ax_t1",
        "brisc2025_test_00352_me_ax_t1",
    ),
]


def load_grayscale(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("L")).copy()


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    for train_name, test_name in PAIRS:
        train_image = DATASET / "segmentation_task" / "train" / "images" / f"{train_name}.jpg"
        test_image = DATASET / "segmentation_task" / "test" / "images" / f"{test_name}.jpg"

        train_mask = DATASET / "segmentation_task" / "train" / "masks" / f"{train_name}.png"
        test_mask = DATASET / "segmentation_task" / "test" / "masks" / f"{test_name}.png"

        paths = [train_image, train_mask, test_image, test_mask]

        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            print("Missing files:")
            for path in missing:
                print(" ", path)
            continue

        image_a = load_grayscale(train_image)
        mask_a = load_grayscale(train_mask)
        image_b = load_grayscale(test_image)
        mask_b = load_grayscale(test_mask)

        if image_a.shape != image_b.shape:
            print(f"Image dimensions differ: {train_name}")
            continue

        # Display binary foreground only for visualization.
        # The correct training threshold must be confirmed separately.
        binary_a = mask_a > 0
        binary_b = mask_b > 0

        fig, axes = plt.subplots(2, 3, figsize=(12, 8))

        axes[0, 0].imshow(image_a, cmap="gray")
        axes[0, 0].set_title("Training MRI")

        axes[0, 1].imshow(mask_a, cmap="gray", vmin=0, vmax=255)
        axes[0, 1].set_title("Training mask (original values)")

        axes[0, 2].imshow(image_a, cmap="gray")
        axes[0, 2].imshow(binary_a, cmap="autumn", alpha=0.45)
        axes[0, 2].set_title("Training overlay")

        axes[1, 0].imshow(image_b, cmap="gray")
        axes[1, 0].set_title("Test MRI")

        axes[1, 1].imshow(mask_b, cmap="gray", vmin=0, vmax=255)
        axes[1, 1].set_title("Test mask (original values)")

        axes[1, 2].imshow(image_b, cmap="gray")
        axes[1, 2].imshow(binary_b, cmap="autumn", alpha=0.45)
        axes[1, 2].set_title("Test overlay")

        for axis in axes.flat:
            axis.axis("off")

        fig.suptitle(f"Duplicate MRI mask comparison: {train_name}")
        fig.tight_layout()

        output_file = OUTPUT / f"{train_name}_comparison.png"
        fig.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()
