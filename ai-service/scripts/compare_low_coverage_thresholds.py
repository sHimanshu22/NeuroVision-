
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "segmentation_manifest.csv"
AUDIT = ROOT / "data" / "mask_quality_audit.json"

THRESHOLDS = [0, 1, 128, 254]
IMAGE_SIZE = 256


def main():
    report = json.loads(AUDIT.read_text(encoding="utf-8"))

    flagged = set(
        report["train"]["low_coverage_filenames"]
        + report["test"]["low_coverage_filenames"]
    )

    rows = []

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

            rows.append(row)

    rows.sort(key=lambda r: (r["split"], r["filename"]))

    print("Threshold comparison for low-coverage masks")
    print("Counts are measured after resizing to 256 x 256.")
    print("Threshold rule: grayscale pixel value > threshold\n")

    for row in rows:
        with Image.open(row["mask_path"]) as source:
            mask = source.convert("L").resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                Image.Resampling.NEAREST,
            )

        array = np.asarray(mask, dtype=np.uint8)

        print(f"\n{row['split'].upper()}: {row['filename']}")
        for threshold in THRESHOLDS:
            foreground = array > threshold
            pixels = int(foreground.sum())
            percentage = 100 * pixels / foreground.size
            print(
                f"  mask > {threshold:3}: "
                f"{pixels:6} pixels ({percentage:.4f}%)"
            )


if __name__ == "__main__":
    main()
