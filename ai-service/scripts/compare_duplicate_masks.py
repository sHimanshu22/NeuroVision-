
import json
from pathlib import Path

import numpy as np
from PIL import Image


DATASET = Path(r"S:\NeuroVision\dataset\archive (4)\brisc2025")
REPORT = Path(__file__).resolve().parents[1] / "data" / "audit_report.json"


def mask_path_from_image(image_path):
    """Map a segmentation image to its corresponding PNG mask."""
    image_path = Path(image_path)
    split = image_path.parent.parent.name

    return (
        DATASET
        / "segmentation_task"
        / split
        / "masks"
        / f"{image_path.stem}.png"
    )


def load_mask(path):
    """Load a mask without silently changing its original dimensions."""
    with Image.open(path) as image:
        original_mode = image.mode
        original_size = image.size
        pixels = np.asarray(image).copy()

    # Convert to grayscale for an additional comparable representation.
    with Image.open(path) as image:
        grayscale = np.asarray(image.convert("L")).copy()

    return original_mode, original_size, pixels, grayscale


def compare_masks(path_a, path_b):
    result = {
        "mask_a": str(path_a),
        "mask_b": str(path_b),
    }

    if not path_a.is_file() or not path_b.is_file():
        result["error"] = "One or both masks are missing."
        return result

    mode_a, size_a, raw_a, gray_a = load_mask(path_a)
    mode_b, size_b, raw_b, gray_b = load_mask(path_b)

    result["mode_a"] = mode_a
    result["mode_b"] = mode_b
    result["size_a"] = list(size_a)
    result["size_b"] = list(size_b)

    if size_a != size_b:
        result["error"] = "Mask dimensions differ."
        return result

    result["raw_array_shapes_equal"] = raw_a.shape == raw_b.shape

    if raw_a.shape == raw_b.shape:
        result["raw_pixels_identical"] = bool(
            np.array_equal(raw_a, raw_b)
        )
    else:
        result["raw_pixels_identical"] = False

    result["grayscale_pixels_identical"] = bool(
        np.array_equal(gray_a, gray_b)
    )

    differing_pixels = int(np.count_nonzero(gray_a != gray_b))
    total_pixels = int(gray_a.size)

    foreground_a = gray_a > 0
    foreground_b = gray_b > 0

    count_a = int(np.count_nonzero(foreground_a))
    count_b = int(np.count_nonzero(foreground_b))
    intersection = int(np.count_nonzero(foreground_a & foreground_b))
    union = int(np.count_nonzero(foreground_a | foreground_b))

    result["different_grayscale_pixels"] = differing_pixels
    result["total_pixels"] = total_pixels
    result["different_pixel_percentage"] = round(
        100 * differing_pixels / total_pixels, 6
    )
    result["grayscale_values_a"] = sorted(
        int(value) for value in np.unique(gray_a)
    )
    result["grayscale_values_b"] = sorted(
        int(value) for value in np.unique(gray_b)
    )
    result["foreground_pixels_a"] = count_a
    result["foreground_pixels_b"] = count_b
    result["foreground_iou"] = (
        round(intersection / union, 6) if union else None
    )
    result["both_masks_empty"] = union == 0

    return result


def main():
    if not REPORT.is_file():
        raise SystemExit(f"Audit report not found: {REPORT}")

    report = json.loads(REPORT.read_text(encoding="utf-8"))

    duplicates = report["segmentation"][
        "train_test_exact_image_duplicates"
    ]

    results = []

    print("=" * 70)
    print("NEUROVISION - DUPLICATE MASK PIXEL COMPARISON")
    print("=" * 70)

    for group_number, item in enumerate(duplicates, start=1):
        train_images = item["train_files"]
        test_images = item["test_files"]

        print(f"\nGroup {group_number}: {item['sha256']}")

        for train_image in train_images:
            for test_image in test_images:
                train_mask = mask_path_from_image(train_image)
                test_mask = mask_path_from_image(test_image)

                comparison = compare_masks(train_mask, test_mask)
                comparison["image_sha256"] = item["sha256"]
                comparison["train_image"] = train_image
                comparison["test_image"] = test_image
                results.append(comparison)

                print(f"  Train mask: {train_mask.name}")
                print(f"  Test mask:  {test_mask.name}")

                if "error" in comparison:
                    print(f"  ERROR: {comparison['error']}")
                    continue

                print(
                    "  Raw pixels identical:",
                    comparison["raw_pixels_identical"],
                )
                print(
                    "  Grayscale pixels identical:",
                    comparison["grayscale_pixels_identical"],
                )
                print(
                    "  Different pixels (%):",
                    comparison["different_pixel_percentage"],
                )
                print(
                    "  Foreground pixels:",
                    comparison["foreground_pixels_a"],
                    "vs",
                    comparison["foreground_pixels_b"],
                )
                print(
                    "  Foreground IoU:",
                    comparison["foreground_iou"],
                )

    output = REPORT.parent / "duplicate_mask_comparison.json"
    output.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print(f"\nDetailed comparison saved to:\n{output}")


if __name__ == "__main__":
    main()
