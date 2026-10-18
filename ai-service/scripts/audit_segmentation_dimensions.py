
from collections import Counter
from pathlib import Path
import json

from PIL import Image

DATASET = Path(r"S:\NeuroVision\dataset\archive (4)\brisc2025")
TASK = DATASET / "segmentation_task"

SPLITS = ["train", "test"]


def image_size(path):
    with Image.open(path) as image:
        image.verify()

    with Image.open(path) as image:
        return image.size  # (width, height)


def main():
    report = {}

    for split in SPLITS:
        image_dir = TASK / split / "images"
        mask_dir = TASK / split / "masks"

        image_sizes = Counter()
        mask_sizes = Counter()
        mismatches = []
        missing_masks = []
        corrupt_files = []

        for image_path in sorted(image_dir.glob("*.jpg")):
            mask_path = mask_dir / f"{image_path.stem}.png"

            try:
                image_dimensions = image_size(image_path)
                image_sizes[image_dimensions] += 1

                if not mask_path.is_file():
                    missing_masks.append(image_path.name)
                    continue

                mask_dimensions = image_size(mask_path)
                mask_sizes[mask_dimensions] += 1

                if image_dimensions != mask_dimensions:
                    mismatches.append({
                        "image": image_path.name,
                        "image_size": image_dimensions,
                        "mask_size": mask_dimensions,
                    })

            except Exception as error:
                corrupt_files.append({
                    "file": image_path.name,
                    "error": str(error),
                })

        report[split] = {
            "image_count": sum(image_sizes.values()),
            "mask_count": sum(mask_sizes.values()),
            "image_dimensions": {
                f"{w}x{h}": count
                for (w, h), count in image_sizes.most_common()
            },
            "mask_dimensions": {
                f"{w}x{h}": count
                for (w, h), count in mask_sizes.most_common()
            },
            "missing_masks": missing_masks,
            "dimension_mismatches": mismatches,
            "corrupt_files": corrupt_files,
        }

        print(f"\n--- {split.upper()} ---")
        print("Images:", report[split]["image_count"])
        print("Masks:", report[split]["mask_count"])
        print("Image dimensions:", report[split]["image_dimensions"])
        print("Mask dimensions:", report[split]["mask_dimensions"])
        print("Missing masks:", len(missing_masks))
        print("Dimension mismatches:", len(mismatches))
        print("Corrupt files:", len(corrupt_files))

    output = Path(__file__).resolve().parents[1] / "data"
    output.mkdir(parents=True, exist_ok=True)

    output_file = output / "segmentation_dimensions_report.json"
    output_file.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nReport saved to: {output_file}")


if __name__ == "__main__":
    main()
