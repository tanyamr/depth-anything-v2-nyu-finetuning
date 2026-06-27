"""PyTorch Dataset for NYU Depth V2.

The dataset reads split files from:
    data/processed/train.txt
    data/processed/val.txt
    data/processed/test.txt

Each line must contain:
    image_path depth_path
"""

from pathlib import Path
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter
import torchvision.transforms.functional as TF


class NYUDepthDataset(Dataset):
    """Dataset that returns RGB image, depth map and valid depth mask."""

    def __init__(
        self,
        split="train",
        image_size=(384, 384),
        min_depth=0.001,
        max_depth=10.0,
        processed_dir="data/processed",
        split_file=None,
    ):
        if split not in ("train", "val", "test"):
            raise ValueError("split must be one of: train, val, test")

        self.split = split
        self.image_size = image_size
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.processed_dir = Path(processed_dir)

        if split_file is None:
            split_file = self.processed_dir / f"{split}.txt"
        else:
            split_file = Path(split_file)

        self.samples = self._read_split_file(split_file)

        # A small and simple augmentation for training images only.
        self.color_jitter = ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05,
        )

    def _read_split_file(self, split_file):
        """Read image-depth path pairs from a split file."""
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        samples = []

        with split_file.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue

                image_path, depth_path = line.split()
                samples.append((Path(image_path), Path(depth_path)))

        if not samples:
            raise ValueError(f"Split file is empty: {split_file}")

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, depth_path = self.samples[index]

        image = self._load_image(image_path)
        depth = self._load_depth(depth_path)

        image, depth = self._resize(image, depth)

        if self.split == "train":
            image, depth = self._augment(image, depth)

        image = self._normalize_image(image)
        depth = torch.from_numpy(depth).float().unsqueeze(0)

        valid_mask = torch.isfinite(depth)
        valid_mask = valid_mask & (depth >= self.min_depth) & (depth <= self.max_depth)

        return image, depth, valid_mask

    def _load_image(self, image_path):
        """Load RGB image as PIL Image."""
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        return Image.open(image_path).convert("RGB")

    def _load_depth(self, depth_path):
        """Load depth map from .npy file."""
        if not depth_path.exists():
            raise FileNotFoundError(f"Depth file not found: {depth_path}")

        return np.load(depth_path).astype(np.float32)

    def _resize(self, image, depth):
        """Resize image and depth map to the target size."""
        height, width = self.image_size

        image = image.resize((width, height), Image.BILINEAR)

        depth_image = Image.fromarray(depth)
        depth_image = depth_image.resize((width, height), Image.NEAREST)
        depth = np.array(depth_image, dtype=np.float32)

        return image, depth

    def _augment(self, image, depth):
        """Apply simple training augmentations."""
        if random.random() < 0.5:
            image = TF.hflip(image)
            depth = np.fliplr(depth).copy()

        image = self.color_jitter(image)

        return image, depth

    def _normalize_image(self, image):
        """Convert RGB image to normalized FloatTensor [3, H, W]."""
        image = TF.to_tensor(image)

        # Standard ImageNet normalization used by many pretrained models.
        image = TF.normalize(
            image,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        return image
