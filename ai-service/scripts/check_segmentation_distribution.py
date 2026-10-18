
import csv
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "segmentation_manifest.csv"

eligible = []

with MANIFEST.open("r", newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)

    for row in reader:
        if row["split"].strip().lower() != "train":
            continue

        if row["included_for_training"].strip().lower() == "true":
            eligible.append(row)

random.Random(42).shuffle(eligible)

validation_count = round(len(eligible) * 0.10)
validation = eligible[:validation_count]
training = eligible[validation_count:]


def tumor_type(row):
    # Example filename: brisc2025_train_00001_gl_ax_t1.jpg
    parts = Path(row["filename"]).stem.split("_")
    code = parts[-3]

    return {
        "gl": "glioma",
        "me": "meningioma",
        "pi": "pituitary",
    }.get(code, f"unknown ({code})")


def report(name, rows):
    counts = Counter(tumor_type(row) for row in rows)

    print(f"\n{name}: {len(rows)} samples")

    for label in sorted(counts):
        count = counts[label]
        percentage = count / len(rows) * 100
        print(f"  {label:15} {count:4} ({percentage:5.2f}%)")


report("TRAINING", training)
report("VALIDATION", validation)

print("\nDistribution check complete.")
