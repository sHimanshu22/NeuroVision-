
import csv
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "segmentation_manifest.csv"
OUTPUT = ROOT / "outputs" / "segmentation" / "resize_comparison_256_384.csv"

THRESHOLD = 128


def main():
    with MANIFEST.open("r", newline="", encoding="utf-8") as f:
        rows = [
            row for row in csv.DictReader(f)
            if row["split"] == "train"
        ]

    results = []

    for row in rows:
        with Image.open(row["mask_path"]) as source:
            original = source.convert("L")
            original_array = np.asarray(original) >= THRESHOLD
            original_pixels = int(original_array.sum())

            counts = {}
            for size in (256, 384):
                resized = original.resize(
                    (size, size),
                    resample=Image.Resampling.NEAREST,
                )
                resized_array = np.asarray(resized) >= THRESHOLD
                counts[size] = int(resized_array.sum())

        results.append({
            "filename": Path(row["filename"]).name,
            "original_width": original.width,
            "original_height": original.height,
            "original_pixels": original_pixels,
            "pixels_256": counts[256],
            "pixels_384": counts[384],
            "retention_256": (
                counts[256] / original_pixels if original_pixels else 0
            ),
            "retention_384": (
                counts[384] / original_pixels if original_pixels else 0
            ),
            "empty_256": counts[256] == 0,
            "empty_384": counts[384] == 0,
        })

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print("Training masks:", len(results))
    print("Empty at 256:", sum(r["empty_256"] for r in results))
    print("Empty at 384:", sum(r["empty_384"] for r in results))

    tiny = sorted(
        results,
        key=lambda r: r["original_pixels"],
    )[:15]

    print("\nSmallest original masks:")
    for r in tiny:
        print(
            f"{r['filename']} | original={r['original_pixels']} | "
            f"256={r['pixels_256']} | 384={r['pixels_384']}"
        )

    print("\nSaved:", OUTPUT)


if __name__ == "__main__":
    main()
