
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "segmentation_manifest.csv"
AUDIT = ROOT / "data" / "mask_quality_audit.json"
OUTPUT = ROOT / "data" / "low_coverage_mask_review.png"

IMAGE_SIZE = 256
THRESHOLD = 128


def main():
    report = json.loads(AUDIT.read_text(encoding="utf-8"))

    flagged = set(
        report["train"]["low_coverage_filenames"]
        + report["test"]["low_coverage_filenames"]
    )

    samples = []

    with MANIFEST.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        for row in csv.DictReader(file):
            if row["filename"] not in flagged:
                continue

            if (
                row["split"] == "train"
                and row["included_for_training"].strip().lower()
                != "true"
            ):
                continue

            samples.append(row)

    samples.sort(key=lambda row: (row["split"], row["filename"]))

    if len(samples) != len(flagged):
        raise RuntimeError(
            f"Expected {len(flagged)} flagged samples; "
            f"found {len(samples)} in the manifest."
        )

    fig, axes = plt.subplots(
        len(samples), 3, figsize=(12, 3.4 * len(samples))
    )

    for i, row in enumerate(samples):
        with Image.open(row["image_path"]) as source:
            image = source.convert("L").resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                Image.Resampling.BILINEAR,
            )

        with Image.open(row["mask_path"]) as source:
            original_mask = source.convert("L")

        mask = original_mask.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.NEAREST,
        )

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        mask_array = np.asarray(mask, dtype=np.uint8)
        binary = mask_array >= THRESHOLD
        coverage = binary.mean() * 100

        axes[i, 0].imshow(image_array, cmap="gray")
        axes[i, 0].set_title(
            f"{row['split'].upper()} MRI\n{row['filename']}",
            fontsize=8,
        )

        axes[i, 1].imshow(binary, cmap="gray", vmin=0, vmax=1)
        axes[i, 1].set_title(
            f"Mask at threshold {THRESHOLD}\n"
            f"Foreground: {coverage:.4f}%",
            fontsize=9,
        )

        axes[i, 2].imshow(image_array, cmap="gray")
        axes[i, 2].imshow(
            np.ma.masked_where(~binary, binary),
            cmap="autumn",
            alpha=0.6,
            vmin=0,
            vmax=1,
        )
        axes[i, 2].set_title("Overlay", fontsize=9)

        for ax in axes[i]:
            ax.axis("off")

    fig.suptitle(
        "NeuroVision — Review of Low-Coverage Masks",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(OUTPUT, dpi=140, bbox_inches="tight")
    plt.close(fig)

    print("Samples reviewed:", len(samples))
    print("Preview saved to:", OUTPUT)


if __name__ == "__main__":
    main()
