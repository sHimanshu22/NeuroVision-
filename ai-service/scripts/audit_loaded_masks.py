
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "segmentation_manifest.csv"
OUTPUT = ROOT / "data" / "mask_quality_audit.json"

IMAGE_SIZE = 256
MASK_THRESHOLD = 128

results = []
counts = {"train": 0, "test": 0}
errors = []

with MANIFEST.open("r", newline="", encoding="utf-8-sig") as file:
    reader = csv.DictReader(file)

    for row in reader:
        split = row["split"].strip().lower()

        if split == "train" and (
            row["included_for_training"].strip().lower() != "true"
        ):
            continue

        if split not in ("train", "test"):
            continue

        counts[split] += 1
        filename = row["filename"]

        try:
            with Image.open(row["mask_path"]) as mask_file:
                mask = mask_file.convert("L")

            original_size = mask.size
            mask = mask.resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                resample=Image.Resampling.NEAREST,
            )

            values = np.asarray(mask, dtype=np.uint8)
            binary = values >= MASK_THRESHOLD
            coverage = float(binary.mean())

            results.append({
                "split": split,
                "filename": filename,
                "original_width": original_size[0],
                "original_height": original_size[1],
                "foreground_fraction": coverage,
                "foreground_pixels": int(binary.sum()),
                "empty": not bool(binary.any()),
                "coverage_below_0_1_percent": coverage < 0.001,
                "coverage_above_50_percent": coverage > 0.50,
            })

        except Exception as exc:
            errors.append({
                "filename": filename,
                "error": str(exc),
            })

coverages = [r["foreground_fraction"] for r in results]

def summarize(split):
    values = [
        r["foreground_fraction"]
        for r in results if r["split"] == split
    ]
    empty = [
        r["filename"] for r in results
        if r["split"] == split and r["empty"]
    ]
    low = [
        r["filename"] for r in results
        if r["split"] == split and r["coverage_below_0_1_percent"]
    ]
    high = [
        r["filename"] for r in results
        if r["split"] == split and r["coverage_above_50_percent"]
    ]

    return {
        "samples_audited": len(values),
        "empty_masks": len(empty),
        "coverage_below_0_1_percent": len(low),
        "coverage_above_50_percent": len(high),
        "coverage_percentiles": {
            "min": float(np.min(values)) if values else None,
            "p1": float(np.percentile(values, 1)) if values else None,
            "median": float(np.median(values)) if values else None,
            "p99": float(np.percentile(values, 99)) if values else None,
            "max": float(np.max(values)) if values else None,
        },
        "empty_filenames": empty[:100],
        "low_coverage_filenames": low[:100],
        "high_coverage_filenames": high[:100],
    }

report = {
    "image_size": IMAGE_SIZE,
    "mask_threshold": MASK_THRESHOLD,
    "counts_selected": counts,
    "samples_successfully_audited": len(results),
    "errors_count": len(errors),
    "errors": errors[:100],
    "train": summarize("train"),
    "test": summarize("test"),
    "note": (
        "Coverage is measured after nearest-neighbor resizing to 256x256 "
        "and thresholding mask grayscale values at >=128. "
        "Outlier cutoffs are review flags, not confirmed errors."
    ),
}

OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

print("Mask quality audit completed.")
print("Selected samples:", counts)
print("Successfully audited:", len(results))
print("Errors:", len(errors))
print("Report:", OUTPUT)

for split in ("train", "test"):
    print(f"\n{split.upper()}")
    print(json.dumps(report[split], indent=2))
