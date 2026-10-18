
import csv
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

MANIFEST = ROOT / "data" / "segmentation_manifest.csv"
COMPARISON = ROOT / "outputs" / "segmentation" / "checkpoint_comparison.csv"
OUTPUT = ROOT / "outputs" / "segmentation" / "worst_case_data_audit.csv"


def main():
    with MANIFEST.open("r", newline="", encoding="utf-8") as f:
        manifest_rows = list(csv.DictReader(f))

    with COMPARISON.open("r", newline="", encoding="utf-8") as f:
        comparison_rows = list(csv.DictReader(f))

    manifest_by_name = {
        Path(row["filename"]).name: row
        for row in manifest_rows
        if row["split"] == "train"
    }

    worst = sorted(
        comparison_rows,
        key=lambda row: float(row["continued_dice"]),
    )[:20]

    results = []

    for row in worst:
        filename = row["filename"]
        manifest_row = manifest_by_name.get(filename)

        if manifest_row is None:
            print("WARNING: Missing manifest entry:", filename)
            continue

        image_path = Path(manifest_row["image_path"])
        mask_path = Path(manifest_row["mask_path"])

        with Image.open(image_path) as image:
            image_array = np.asarray(image.convert("L"))
            image_size = image.size

        with Image.open(mask_path) as mask:
            mask_array = np.asarray(mask.convert("L"))
            mask_size = mask.size

        foreground = mask_array >= 128
        foreground_pixels = int(foreground.sum())
        total_pixels = int(foreground.size)

        results.append({
            "filename": filename,
            "continued_dice": float(row["continued_dice"]),
            "baseline_dice": float(row["baseline_dice"]),
            "image_width": image_size[0],
            "image_height": image_size[1],
            "mask_width": mask_size[0],
            "mask_height": mask_size[1],
            "dimensions_match": image_size == mask_size,
            "mask_foreground_pixels": foreground_pixels,
            "mask_foreground_percentage": (
                100.0 * foreground_pixels / total_pixels
            ),
            "mask_unique_values": len(np.unique(mask_array)),
            "image_path_exists": image_path.exists(),
            "mask_path_exists": mask_path.exists(),
        })

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(results[0].keys()),
        )
        writer.writeheader()
        writer.writerows(results)

    print("Audited cases:", len(results))
    print("Saved report:", OUTPUT)

    for result in results:
        print(
            f"{result['filename']} | "
            f"Dice={result['continued_dice']:.4f} | "
            f"image={result['image_width']}x{result['image_height']} | "
            f"mask={result['mask_width']}x{result['mask_height']} | "
            f"aligned_dims={result['dimensions_match']} | "
            f"foreground={result['mask_foreground_percentage']:.3f}%"
        )


if __name__ == "__main__":
    main()
