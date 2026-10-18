
import csv
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "segmentation_manifest.csv"
OUTPUT = ROOT / "outputs" / "segmentation" / "mask_resize_effect.csv"

TARGET_SIZE = (256, 256)
THRESHOLD = 128


def main():
    with MANIFEST.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    results = []

    for row in rows:
        if row["split"] != "train":
            continue

        filename = Path(row["filename"]).name

        with Image.open(row["mask_path"]) as source:
            original = source.convert("L")
            original_array = np.asarray(original) >= THRESHOLD

            resized = original.resize(
                TARGET_SIZE,
                resample=Image.Resampling.NEAREST,
            )
            resized_array = np.asarray(resized) >= THRESHOLD

        original_pixels = int(original_array.sum())
        resized_pixels = int(resized_array.sum())

        results.append({
            "filename": filename,
            "original_width": original.width,
            "original_height": original.height,
            "original_foreground_pixels": original_pixels,
            "resized_foreground_pixels": resized_pixels,
            "original_foreground_percentage": (
                100.0 * original_pixels / original_array.size
            ),
            "resized_foreground_percentage": (
                100.0 * resized_pixels / resized_array.size
            ),
            "empty_after_resize": resized_pixels == 0,
            "foreground_retention_ratio": (
                resized_pixels / original_pixels
                if original_pixels else 0.0
            ),
        })

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(results[0].keys()),
        )
        writer.writeheader()
        writer.writerows(results)

    print("Training masks audited:", len(results))
    print("Masks empty after resizing:", sum(
        r["empty_after_resize"] for r in results
    ))

    smallest = sorted(
        results,
        key=lambda r: r["original_foreground_pixels"],
    )[:15]

    print("\n15 masks with the fewest original foreground pixels:")
    for r in smallest:
        print(
            f"{r['filename']} | "
            f"original={r['original_foreground_pixels']} px | "
            f"resized={r['resized_foreground_pixels']} px | "
            f"retention={r['foreground_retention_ratio']:.3f}"
        )

    print("\nSaved:", OUTPUT)


if __name__ == "__main__":
    main()
