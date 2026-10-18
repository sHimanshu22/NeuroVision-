
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "segmentation_manifest.csv"
OUTPUT = ROOT / "data" / "mask_threshold_comparison.json"

IMAGE_SIZE = 256


def main():
    summaries = {
        "train": {
            "count": 0,
            "masks_with_intermediate_values": 0,
            "pixels_total": 0,
            "foreground_ge_128": 0,
            "foreground_eq_255": 0,
            "intermediate_pixels": 0,
            "masks_area_changed": 0,
            "empty_ge_128": 0,
            "empty_eq_255": 0,
        },
        "test": {
            "count": 0,
            "masks_with_intermediate_values": 0,
            "pixels_total": 0,
            "foreground_ge_128": 0,
            "foreground_eq_255": 0,
            "intermediate_pixels": 0,
            "masks_area_changed": 0,
            "empty_ge_128": 0,
            "empty_eq_255": 0,
        },
    }

    examples = []
    errors = []

    with MANIFEST.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        for row in csv.DictReader(file):
            split = row["split"].strip().lower()

            if split not in summaries:
                continue

            if (
                split == "train"
                and row["included_for_training"].strip().lower()
                != "true"
            ):
                continue

            try:
                with Image.open(row["mask_path"]) as source:
                    original = np.asarray(
                        source.convert("L"), dtype=np.uint8
                    )

                resized_image = Image.fromarray(original).resize(
                    (IMAGE_SIZE, IMAGE_SIZE),
                    Image.Resampling.NEAREST,
                )
                mask = np.asarray(resized_image, dtype=np.uint8)

                # Compare both rules on the same resized mask.
                ge_128 = mask >= 128
                eq_255 = mask == 255
                intermediate = (mask > 0) & (mask < 255)

                stats = summaries[split]
                stats["count"] += 1
                stats["pixels_total"] += int(mask.size)
                stats["foreground_ge_128"] += int(ge_128.sum())
                stats["foreground_eq_255"] += int(eq_255.sum())
                stats["intermediate_pixels"] += int(intermediate.sum())

                has_intermediate = bool(
                    np.any((original > 0) & (original < 255))
                )
                if has_intermediate:
                    stats["masks_with_intermediate_values"] += 1

                changed = int(ge_128.sum()) != int(eq_255.sum())
                if changed:
                    stats["masks_area_changed"] += 1

                if not ge_128.any():
                    stats["empty_ge_128"] += 1
                if not eq_255.any():
                    stats["empty_eq_255"] += 1

                if changed and len(examples) < 20:
                    examples.append({
                        "split": split,
                        "filename": row["filename"],
                        "original_unique_values": int(
                            np.unique(original).size
                        ),
                        "original_intermediate_pixels": int(
                            ((original > 0) & (original < 255)).sum()
                        ),
                        "resized_foreground_ge_128": int(ge_128.sum()),
                        "resized_foreground_eq_255": int(eq_255.sum()),
                        "resized_pixels_different": int(
                            np.logical_xor(ge_128, eq_255).sum()
                        ),
                    })

            except Exception as exc:
                errors.append({
                    "filename": row["filename"],
                    "split": split,
                    "error": str(exc),
                })

    for stats in summaries.values():
        total = stats["pixels_total"]
        stats["foreground_fraction_ge_128"] = (
            stats["foreground_ge_128"] / total if total else None
        )
        stats["foreground_fraction_eq_255"] = (
            stats["foreground_eq_255"] / total if total else None
        )
        stats["intermediate_pixel_fraction"] = (
            stats["intermediate_pixels"] / total if total else None
        )

    report = {
        "image_size": IMAGE_SIZE,
        "comparison": "mask >= 128 versus mask == 255",
        "counts": summaries,
        "example_masks_with_area_changes": examples,
        "error_count": len(errors),
        "errors": errors[:100],
        "note": (
            "Read-only audit. Masks are converted to grayscale and "
            "nearest-neighbor resized to 256x256. This compares binary "
            "rules; it does not determine which rule is the true label."
        ),
    }

    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Mask threshold comparison completed.")
    print("Report:", OUTPUT)
    print("Errors:", len(errors))

    for split, stats in summaries.items():
        print(f"\n{split.upper()}")
        print("Masks audited:", stats["count"])
        print(
            "Masks with original intermediate values:",
            stats["masks_with_intermediate_values"],
        )
        print("Masks whose foreground area changes:", stats["masks_area_changed"])
        print("Foreground pixels (>=128):", stats["foreground_ge_128"])
        print("Foreground pixels (==255):", stats["foreground_eq_255"])
        print("Intermediate pixels after resize:", stats["intermediate_pixels"])
        print("Empty masks (>=128):", stats["empty_ge_128"])
        print("Empty masks (==255):", stats["empty_eq_255"])
        print(
            "Foreground fraction (>=128):",
            stats["foreground_fraction_ge_128"],
        )
        print(
            "Foreground fraction (==255):",
            stats["foreground_fraction_eq_255"],
        )


if __name__ == "__main__":
    main()
