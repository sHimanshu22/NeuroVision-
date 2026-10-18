
import csv
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


class BrainSegmentationDataset(Dataset):
    """
    Loads MRI images and masks from segmentation_manifest.csv.

    Training and validation are split deterministically from eligible
    training records. Official test records are kept separate.
    """

    def __init__(
        self,
        manifest_path,
        split="train",
        image_size=256,
        mask_threshold=128,
        validation_fraction=0.10,
        seed=42,
    ):
        if split not in ("train", "validation", "test"):
            raise ValueError(
                "split must be 'train', 'validation', or 'test'"
            )

        if not 0 < validation_fraction < 1:
            raise ValueError(
                "validation_fraction must be between 0 and 1"
            )

        self.manifest_path = Path(manifest_path)
        self.image_size = image_size
        self.mask_threshold = mask_threshold
        self.samples = []

        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"Manifest not found: {self.manifest_path}"
            )

        eligible_train = []
        official_test = []

        with self.manifest_path.open(
            "r", newline="", encoding="utf-8-sig"
        ) as file:
            reader = csv.DictReader(file)

            required_columns = {
                "split",
                "filename",
                "image_path",
                "mask_path",
                "included_for_training",
            }

            if not required_columns.issubset(reader.fieldnames or []):
                raise ValueError(
                    "Manifest is missing required columns."
                )

            for row in reader:
                row_split = row["split"].strip().lower()

                if row_split == "train":
                    if (
                        row["included_for_training"].strip().lower()
                        == "true"
                    ):
                        eligible_train.append(row)

                elif row_split == "test":
                    official_test.append(row)

        # Deterministic split: the same manifest and seed always
        # produce the same training and validation filenames.
        shuffled_train = eligible_train.copy()
        random.Random(seed).shuffle(shuffled_train)

        validation_count = round(
            len(shuffled_train) * validation_fraction
        )

        validation_rows = shuffled_train[:validation_count]
        training_rows = shuffled_train[validation_count:]

        if split == "train":
            selected_rows = training_rows
        elif split == "validation":
            selected_rows = validation_rows
        else:
            selected_rows = official_test

        for row in selected_rows:
            image_path = Path(row["image_path"])
            mask_path = Path(row["mask_path"])

            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Image not found: {image_path}"
                )

            if not mask_path.is_file():
                raise FileNotFoundError(
                    f"Mask not found: {mask_path}"
                )

            self.samples.append(
                {
                    "filename": row["filename"],
                    "image_path": image_path,
                    "mask_path": mask_path,
                }
            )

        if not self.samples:
            raise ValueError(
                f"No samples found for split='{split}'"
            )

        print(
            f"Loaded {len(self.samples)} samples "
            f"for split='{split}'"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        with Image.open(sample["image_path"]) as image_file:
            image = image_file.convert("L")

        with Image.open(sample["mask_path"]) as mask_file:
            mask = mask_file.convert("L")

        image = TF.resize(
            image,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
        )

        mask = TF.resize(
            mask,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.NEAREST,
        )

        image_tensor = TF.to_tensor(image)

        mask_array = np.array(mask, dtype=np.uint8, copy=True)
        binary_mask = (
            mask_array >= self.mask_threshold
        ).astype(np.float32)

        mask_tensor = torch.from_numpy(binary_mask).unsqueeze(0)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "filename": sample["filename"],
        }
