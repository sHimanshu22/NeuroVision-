
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from segmentation_dataset import BrainSegmentationDataset


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "segmentation_manifest.csv"
OUTPUT = ROOT / "data" / "segmentation_sample_preview.png"


def main():
    train = BrainSegmentationDataset(
        MANIFEST, split="train", image_size=256, mask_threshold=128
    )
    test = BrainSegmentationDataset(
        MANIFEST, split="test", image_size=256, mask_threshold=128
    )

    # Two training samples and two official test samples.
    selections = [
        (train, 0, "Train: first sample"),
        (train, len(train) // 2, "Train: middle sample"),
        (test, 0, "Test: first sample"),
        (test, len(test) // 2, "Test: middle sample"),
    ]

    fig, axes = plt.subplots(4, 3, figsize=(12, 15))

    for row, (dataset, index, title) in enumerate(selections):
        sample = dataset[index]

        image = sample["image"].squeeze(0).numpy()
        mask = sample["mask"].squeeze(0).numpy()

        axes[row, 0].imshow(image, cmap="gray")
        axes[row, 0].set_title(f"{title}\n{sample['filename']}")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title(
            f"Binary mask ({mask.mean() * 100:.2f}% foreground)"
        )
        axes[row, 1].axis("off")

        axes[row, 2].imshow(image, cmap="gray")
        axes[row, 2].imshow(
            np.ma.masked_where(mask < 0.5, mask),
            cmap="autumn",
            alpha=0.55,
            vmin=0,
            vmax=1,
        )
        axes[row, 2].set_title("Mask overlay")
        axes[row, 2].axis("off")

    fig.suptitle(
        "NeuroVision — MRI Segmentation Sample Check",
        fontsize=15,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"Preview saved to: {OUTPUT}")


if __name__ == "__main__":
    main()
