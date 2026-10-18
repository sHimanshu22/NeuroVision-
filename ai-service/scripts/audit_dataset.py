
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image, UnidentifiedImageError


DEFAULT_DATASET = Path(
    r"S:\NeuroVision\dataset\archive (4)\brisc2025"
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

CLASS_NAMES = {"glioma", "meningioma", "pituitary", "no_tumor"}


def sha256_file(path):
    """Calculate a file hash for exact duplicate detection."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def inspect_image(path, inspect_pixels=False):
    """Inspect an image without changing the original file."""
    result = {
        "path": str(path),
        "valid": False,
    }

    try:
        with Image.open(path) as image:
            image.load()

            result["valid"] = True
            result["format"] = image.format
            result["mode"] = image.mode
            result["width"] = image.width
            result["height"] = image.height

            if inspect_pixels:
                grayscale = image.convert("L")
                values = grayscale.getcolors(
                    maxcolors=256
                )

                if values is not None:
                    result["pixel_values"] = sorted(
                        value for value, count in values
                    )
                else:
                    result["pixel_values"] = (
                        "More than 256 grayscale values"
                    )

    except (
        OSError,
        ValueError,
        UnidentifiedImageError,
    ) as error:
        result["error"] = str(error)

    return result


def find_images(folder):
    """Find supported images recursively."""
    if not folder.exists():
        return []

    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def audit_classification(dataset_root):
    root = dataset_root / "classification_task"
    report = {
        "exists": root.exists(),
        "splits": {},
        "train_test_exact_duplicates": [],
    }

    if not root.exists():
        report["error"] = "classification_task not found"
        return report

    hashes_by_split = {}

    for split in ("train", "test"):
        split_root = root / split
        split_report = {
            "exists": split_root.exists(),
            "classes": {},
            "total_images": 0,
            "invalid_images": [],
            "dimension_counts": {},
        }

        if not split_root.exists():
            report["splits"][split] = split_report
            continue

        dimensions = Counter()
        split_hashes = {}

        for class_name in sorted(CLASS_NAMES):
            class_dir = split_root / class_name
            files = find_images(class_dir)

            class_report = {
                "directory_exists": class_dir.exists(),
                "image_count": len(files),
                "valid_images": 0,
                "invalid_images": [],
            }

            for path in files:
                details = inspect_image(path)

                if not details["valid"]:
                    class_report["invalid_images"].append(
                        details
                    )
                    split_report["invalid_images"].append(
                        details
                    )
                    continue

                class_report["valid_images"] += 1
                dimensions[
                    f'{details["width"]}x{details["height"]}'
                ] += 1

                digest = sha256_file(path)
                split_hashes.setdefault(
                    digest, []
                ).append(str(path))

            split_report["classes"][class_name] = (
                class_report
            )

        split_report["total_images"] = sum(
            item["image_count"]
            for item in split_report["classes"].values()
        )
        split_report["valid_images"] = sum(
            item["valid_images"]
            for item in split_report["classes"].values()
        )
        split_report["dimension_counts"] = dict(dimensions)

        # Detect files in unexpected locations or class folders.
        expected_dirs = {
            (split_root / name).resolve()
            for name in CLASS_NAMES
        }
        unexpected = []

        for path in find_images(split_root):
            if path.parent.resolve() not in expected_dirs:
                unexpected.append(str(path))

        split_report["images_outside_expected_class_folders"] = (
            unexpected
        )

        hashes_by_split[split] = split_hashes
        report["splits"][split] = split_report

    train_hashes = hashes_by_split.get("train", {})
    test_hashes = hashes_by_split.get("test", {})

    for digest in sorted(set(train_hashes) & set(test_hashes)):
        report["train_test_exact_duplicates"].append({
            "sha256": digest,
            "train_files": train_hashes[digest],
            "test_files": test_hashes[digest],
        })

    return report


def audit_segmentation(dataset_root):
    root = dataset_root / "segmentation_task"
    report = {
        "exists": root.exists(),
        "splits": {},
        "train_test_exact_image_duplicates": [],
    }

    if not root.exists():
        report["error"] = "segmentation_task not found"
        return report

    image_hashes = {}

    for split in ("train", "test"):
        split_root = root / split
        images_dir = split_root / "images"
        masks_dir = split_root / "masks"

        images = find_images(images_dir)
        masks = find_images(masks_dir)

        # Match by filename without the extension.
        image_by_name = {}
        mask_by_name = {}

        for path in images:
            image_by_name.setdefault(path.stem, []).append(path)

        for path in masks:
            mask_by_name.setdefault(path.stem, []).append(path)

        image_names = set(image_by_name)
        mask_names = set(mask_by_name)

        split_report = {
            "exists": split_root.exists(),
            "images_directory_exists": images_dir.exists(),
            "masks_directory_exists": masks_dir.exists(),
            "image_count": len(images),
            "mask_count": len(masks),
            "matched_filename_count": len(
                image_names & mask_names
            ),
            "images_without_masks": sorted(
                image_names - mask_names
            ),
            "masks_without_images": sorted(
                mask_names - image_names
            ),
            "duplicate_image_stems": sorted(
                name for name, paths in image_by_name.items()
                if len(paths) > 1
            ),
            "duplicate_mask_stems": sorted(
                name for name, paths in mask_by_name.items()
                if len(paths) > 1
            ),
            "invalid_images": [],
            "invalid_masks": [],
            "image_dimensions": {},
            "mask_dimensions": {},
            "image_mask_dimension_mismatches": [],
            "mask_pixel_values": {},
        }

        image_dimensions = Counter()
        mask_dimensions = Counter()
        pixel_value_counts = Counter()
        split_hashes = {}

        # Inspect all images.
        for path in images:
            details = inspect_image(path)

            if not details["valid"]:
                split_report["invalid_images"].append(details)
                continue

            image_dimensions[
                f'{details["width"]}x{details["height"]}'
            ] += 1

            split_hashes.setdefault(
                sha256_file(path), []
            ).append(str(path))

        # Inspect masks and compare paired dimensions.
        for path in masks:
            details = inspect_image(path, inspect_pixels=True)

            if not details["valid"]:
                split_report["invalid_masks"].append(details)
                continue

            mask_dimensions[
                f'{details["width"]}x{details["height"]}'
            ] += 1

            pixel_values = details.get("pixel_values", [])
            if isinstance(pixel_values, list):
                pixel_value_counts.update(pixel_values)

            paired_images = image_by_name.get(path.stem, [])

            for image_path in paired_images:
                image_details = inspect_image(image_path)

                if (
                    image_details["valid"]
                    and (
                        image_details["width"] != details["width"]
                        or image_details["height"] != details["height"]
                    )
                ):
                    split_report[
                        "image_mask_dimension_mismatches"
                    ].append({
                        "image": str(image_path),
                        "mask": str(path),
                        "image_size": [
                            image_details["width"],
                            image_details["height"],
                        ],
                        "mask_size": [
                            details["width"],
                            details["height"],
                        ],
                    })

        split_report["image_dimensions"] = dict(image_dimensions)
        split_report["mask_dimensions"] = dict(mask_dimensions)
        split_report["mask_pixel_value_counts"] = {
            str(key): value
            for key, value in sorted(pixel_value_counts.items())
        }

        image_hashes[split] = split_hashes
        report["splits"][split] = split_report

    train_hashes = image_hashes.get("train", {})
    test_hashes = image_hashes.get("test", {})

    for digest in sorted(set(train_hashes) & set(test_hashes)):
        report["train_test_exact_image_duplicates"].append({
            "sha256": digest,
            "train_files": train_hashes[digest],
            "test_files": test_hashes[digest],
        })

    return report


def audit_manifest(dataset_root):
    report = {
        "csv_exists": (dataset_root / "manifest.csv").exists(),
        "json_exists": (dataset_root / "manifest.json").exists(),
    }

    csv_path = dataset_root / "manifest.csv"

    if csv_path.exists():
        try:
            with csv_path.open(
                "r", encoding="utf-8-sig", newline=""
            ) as file:
                reader = csv.DictReader(file)
                report["csv_columns"] = reader.fieldnames
                report["csv_row_count"] = sum(
                    1 for _ in reader
                )
        except (OSError, csv.Error) as error:
            report["csv_error"] = str(error)

    json_path = dataset_root / "manifest.json"

    if json_path.exists():
        try:
            with json_path.open(
                "r", encoding="utf-8"
            ) as file:
                data = json.load(file)

            report["json_valid"] = True
            report["json_top_level_type"] = type(data).__name__

            if isinstance(data, list):
                report["json_record_count"] = len(data)
            elif isinstance(data, dict):
                report["json_top_level_keys"] = list(data.keys())

        except (OSError, json.JSONDecodeError) as error:
            report["json_valid"] = False
            report["json_error"] = str(error)

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Audit the BRISC2025 MRI dataset."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to the extracted brisc2025 directory",
    )
    args = parser.parse_args()

    dataset_root = args.dataset.resolve()

    if not dataset_root.is_dir():
        raise SystemExit(
            f"Dataset directory not found: {dataset_root}\n"
            "Pass the correct path using --dataset."
        )

    print("=" * 65)
    print("NEUROVISION - BRISC2025 DATASET AUDIT")
    print("=" * 65)
    print(f"Dataset: {dataset_root}")
    print("\nAuditing classification images...")
    classification = audit_classification(dataset_root)

    print("Auditing segmentation images and masks...")
    segmentation = audit_segmentation(dataset_root)

    print("Inspecting dataset manifests...")
    manifest = audit_manifest(dataset_root)

    report = {
        "dataset_root": str(dataset_root),
        "classification": classification,
        "segmentation": segmentation,
        "manifest": manifest,
    }

    output_dir = Path(__file__).resolve().parents[1] / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "audit_report.json"

    output_file.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 65)
    print("AUDIT SUMMARY")
    print("=" * 65)

    for split, info in classification.get("splits", {}).items():
        print(
            f"Classification {split}: "
            f'{info.get("total_images", 0)} images; '
            f'{info.get("valid_images", 0)} valid'
        )

        for class_name, class_info in info.get("classes", {}).items():
            print(
                f'  {class_name}: '
                f'{class_info["image_count"]} images'
            )

    for split, info in segmentation.get("splits", {}).items():
        print(
            f"Segmentation {split}: "
            f'{info.get("image_count", 0)} images, '
            f'{info.get("mask_count", 0)} masks, '
            f'{info.get("matched_filename_count", 0)} matched names'
        )
        print(
            f'  Images without masks: '
            f'{len(info.get("images_without_masks", []))}'
        )
        print(
            f'  Masks without images: '
            f'{len(info.get("masks_without_images", []))}'
        )
        print(
            f'  Dimension mismatches: '
            f'{len(info.get("image_mask_dimension_mismatches", []))}'
        )

    print(
        "Classification train/test exact duplicates:",
        len(classification.get("train_test_exact_duplicates", [])),
    )
    print(
        "Segmentation train/test exact image duplicates:",
        len(segmentation.get("train_test_exact_image_duplicates", [])),
    )
    print(f"\nFull report saved to:\n{output_file}")
    print("\nAudit completed. Review the report before training.")


if __name__ == "__main__":
    main()
