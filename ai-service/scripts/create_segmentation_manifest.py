
from pathlib import Path
from collections import defaultdict
import csv
import hashlib
import json

DATASET = Path(r"S:\NeuroVision\dataset\archive (4)\brisc2025")
TASK = DATASET / "segmentation_task"
OUTPUT = Path(__file__).resolve().parents[1] / "data"

SPLITS = ("train", "test")


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    records = []
    hashes_by_split = {
        "train": defaultdict(list),
        "test": defaultdict(list),
    }

    # Audit image files and their matching masks.
    for split in SPLITS:
        image_dir = TASK / split / "images"
        mask_dir = TASK / split / "masks"

        for image_path in sorted(image_dir.glob("*.jpg")):
            mask_path = mask_dir / f"{image_path.stem}.png"

            if not mask_path.is_file():
                raise FileNotFoundError(
                    f"Missing mask for {split}/{image_path.name}: "
                    f"{mask_path}"
                )

            image_hash = sha256_file(image_path)
            hashes_by_split[split][image_hash].append(image_path.name)

            records.append({
                "split": split,
                "filename": image_path.name,
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "sha256": image_hash,
            })

    # Any training image with a matching test hash is excluded
    # from the leakage-controlled training manifest.
    test_hashes = set(hashes_by_split["test"])

    for record in records:
        matches_test = record["sha256"] in test_hashes

        record["included_for_training"] = (
            record["split"] == "train" and not matches_test
        )

        if record["split"] == "test":
            record["exclusion_reason"] = ""
        elif matches_test:
            record["exclusion_reason"] = (
                "Exact image hash also occurs in official test split"
            )
        else:
            record["exclusion_reason"] = ""

    # Save a complete audit manifest.
    manifest_path = OUTPUT / "segmentation_manifest.csv"

    fields = [
        "split",
        "filename",
        "image_path",
        "mask_path",
        "sha256",
        "included_for_training",
        "exclusion_reason",
    ]

    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    train_records = [r for r in records if r["split"] == "train"]
    test_records = [r for r in records if r["split"] == "test"]

    included_train = [
        r for r in train_records if r["included_for_training"]
    ]
    excluded_train = [
        r for r in train_records
        if not r["included_for_training"]
    ]

    duplicate_groups = {
        image_hash: {
            "train_filenames": train_names,
            "test_filenames": hashes_by_split["test"][image_hash],
        }
        for image_hash, train_names in hashes_by_split["train"].items()
        if image_hash in test_hashes
    }

    summary = {
        "method": "SHA-256 exact file hash comparison",
        "original_dataset_modified": False,
        "train_images_total": len(train_records),
        "test_images_total": len(test_records),
        "train_images_included": len(included_train),
        "train_images_excluded": len(excluded_train),
        "test_images_preserved": len(test_records),
        "duplicate_hash_groups": len(duplicate_groups),
        "excluded_training_filenames": [
            r["filename"] for r in excluded_train
        ],
        "duplicate_groups": duplicate_groups,
    }

    summary_path = OUTPUT / "segmentation_manifest_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n--- SEGMENTATION MANIFEST ---")
    print("Training images:", len(train_records))
    print("Official test images:", len(test_records))
    print("Training images included:", len(included_train))
    print("Training images excluded:", len(excluded_train))
    print("Duplicate hash groups:", len(duplicate_groups))
    print("Official test images preserved:", len(test_records))
    print("Original dataset modified: No")
    print(f"\nManifest: {manifest_path}")
    print(f"Summary:  {summary_path}")

    if len(train_records) != 3933 or len(test_records) != 860:
        raise RuntimeError(
            "Unexpected dataset counts. Review the dataset before proceeding."
        )


if __name__ == "__main__":
    main()
