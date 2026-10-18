
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "segmentation_manifest.csv"
COMPARISON = ROOT / "outputs" / "segmentation" / "checkpoint_comparison.csv"
OUTPUT = ROOT / "outputs" / "segmentation" / "worst_original_mask_audit.png"


def main():
    with MANIFEST.open("r", newline="", encoding="utf-8") as f:
        manifest = list(csv.DictReader(f))

    with COMPARISON.open("r", newline="", encoding="utf-8") as f:
        comparisons = list(csv.DictReader(f))

    manifest_by_name = {
        Path(row["filename"]).name: row
        for row in manifest
        if row["split"] == "train"
    }

    worst = sorted(
        comparisons,
        key=lambda row: float(row["continued_dice"]),
    )[:12]

    fig, axes = plt.subplots(12, 3, figsize=(12, 32))

    for i, record in enumerate(worst):
        filename = record["filename"]
        row = manifest_by_name[filename]

        with Image.open(row["image_path"]) as source:
            image = np.asarray(source.convert("L"))

        with Image.open(row["mask_path"]) as source:
            mask = np.asarray(source.convert("L")) >= 128

        axes[i, 0].imshow(image, cmap="gray")
        axes[i, 0].set_title(filename, fontsize=8)

        axes[i, 1].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axes[i, 1].set_title(
            f"Original mask | foreground={mask.mean() * 100:.3f}%"
        )

        axes[i, 2].imshow(image, cmap="gray")
        overlay = np.ma.masked_where(~mask, mask)
        axes[i, 2].imshow(overlay, cmap="autumn", alpha=0.65, vmin=0, vmax=1)
        axes[i, 2].set_title(
            f"Overlay | continued Dice={float(record['continued_dice']):.3f}"
        )

        for ax in axes[i]:
            ax.axis("off")

    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=140, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
