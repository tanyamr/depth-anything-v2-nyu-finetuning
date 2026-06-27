"""Prepare NYU Depth V2 labeled dataset.

The script reads:
    data/raw/nyu_depth_v2_labeled.mat

It saves:
    data/processed/images/*.png
    data/processed/depths/*.npy
    data/processed/train.txt
    data/processed/val.txt
    data/processed/test.txt
    data/processed/dataset_stats.json
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image


RAW_MAT_PATH = Path("data/raw/nyu_depth_v2_labeled.mat")
PROCESSED_DIR = Path("data/processed")
IMAGES_DIR = PROCESSED_DIR / "images"
DEPTHS_DIR = PROCESSED_DIR / "depths"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
TEST_RATIO = 0.20
SEED = 42


def load_mat_file(mat_path):
    """Load images and depths from the NYU Depth V2 .mat file."""
    if not mat_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {mat_path}")

    try:
        # Works for normal MATLAB .mat files
        from scipy.io import loadmat

        try:
            data = loadmat(mat_path)
            images = data["images"]
            depths = data["depths"]
            return images, depths
        except NotImplementedError:
            pass
        except KeyError as error:
            raise KeyError(
                "The .mat file must contain variables named 'images' and 'depths'"
            ) from error
    except ImportError:
        pass

    # Many NYU Depth V2 files are MATLAB v7.3 files, which use HDF5.
    try:
        import h5py
    except ImportError as error:
        raise ImportError(
            "Could not read the .mat file because neither scipy nor h5py is installed. "
            "Install project dependencies with: pip install -r requirements.txt"
        ) from error

    with h5py.File(mat_path, "r") as file:
        images = np.array(file["images"])
        depths = np.array(file["depths"])
    return images, depths


def find_sample_axis(array, channel_axis=None):
    """Find the axis that stores image/depth index."""
    axes = list(range(array.ndim))
    if channel_axis is not None:
        axes.remove(channel_axis)

    # NYU images are 480x640, so the remaining non-image axis is N
    for axis in axes:
        if array.shape[axis] not in (480, 640):
            return axis

    # Fallback for unusual small test files
    return axes[0]


def normalize_images(images):
    """Convert images to shape (N, H, W, 3) from (N,3,H,W)
        N — количество изображений;
        H — высота;
        W — ширина;
        3 — RGB-каналы"""
    if images.ndim != 4:
        raise ValueError(f"Expected images to have 4 dimensions, got {images.shape}")

    channel_axes = [axis for axis, size in enumerate(images.shape) if size == 3]
    if not channel_axes:
        raise ValueError(f"Could not find RGB channel axis in images: {images.shape}")

    channel_axis = channel_axes[0]
    sample_axis = find_sample_axis(images, channel_axis=channel_axis)

    # First make the array (N, 3, H, W), then convert it to (N, H, W, 3).
    images = np.moveaxis(images, [sample_axis, channel_axis], [0, 1])
    images = np.transpose(images, (0, 2, 3, 1))

    # HDF5 MATLAB files often store images as (N, 3, 640, 480).
    # After transpose this becomes (N, 640, 480, 3), so rotate to 480x640.
    if images.shape[1] == 640 and images.shape[2] == 480:
        images = np.transpose(images, (0, 2, 1, 3))

    return images.astype(np.uint8)


def normalize_depths(depths):
    """Convert depths to shape (N, H, W)."""
    if depths.ndim != 3:
        raise ValueError(f"Expected depths to have 3 dimensions, got {depths.shape}")

    sample_axis = find_sample_axis(depths)
    depths = np.moveaxis(depths, sample_axis, 0)

    # HDF5 MATLAB files often store depth maps as (N, 640, 480).
    if depths.shape[1] == 640 and depths.shape[2] == 480:
        depths = np.transpose(depths, (0, 2, 1))

    return depths.astype(np.float32)


def save_dataset(images, depths):
    """Save RGB images as PNG files and depth maps as NumPy files."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    DEPTHS_DIR.mkdir(parents=True, exist_ok=True)

    pairs = []

    for index in range(len(images)):
        image_path = IMAGES_DIR / f"{index:06d}.png"
        depth_path = DEPTHS_DIR / f"{index:06d}.npy"

        Image.fromarray(images[index]).save(image_path)
        np.save(depth_path, depths[index])

        pairs.append((image_path.as_posix(), depth_path.as_posix()))

    return pairs


def split_dataset(pairs):
    """Split dataset into train, validation and test parts."""
    rng = np.random.default_rng(SEED)
    indices = np.arange(len(pairs))
    rng.shuffle(indices)

    train_count = int(len(pairs) * TRAIN_RATIO)
    val_count = int(len(pairs) * VAL_RATIO)
    test_count = len(pairs) - train_count - val_count

    train_indices = indices[:train_count]
    val_indices = indices[train_count : train_count + val_count]
    test_indices = indices[train_count + val_count :]

    splits = {
        "train": [pairs[index] for index in train_indices],
        "val": [pairs[index] for index in val_indices],
        "test": [pairs[index] for index in test_indices],
    }

    return splits, train_count, val_count, test_count


def write_split_file(split_name, pairs):
    """Write one split file with lines: image_path depth_path."""
    split_path = PROCESSED_DIR / f"{split_name}.txt"

    with split_path.open("w", encoding="utf-8") as file:
        for image_path, depth_path in pairs:
            file.write(f"{image_path} {depth_path}\n")


def calculate_stats(depths, train_count, val_count, test_count):
    """Calculate simple dataset statistics."""
    # Check NaN, Inf
    finite_mask = np.isfinite(depths)
    invalid_count = int(depths.size - np.count_nonzero(finite_mask))
    valid_depths = depths[finite_mask]

    if valid_depths.size == 0:
        min_depth = None
        max_depth = None
        mean_depth = None
    else:
        min_depth = float(np.min(valid_depths))
        max_depth = float(np.max(valid_depths))
        mean_depth = float(np.mean(valid_depths))

    return {
        "num_images": int(len(depths)),
        "num_train": int(train_count),
        "num_val": int(val_count),
        "num_test": int(test_count),
        "min_depth": min_depth,
        "max_depth": max_depth,
        "mean_depth": mean_depth,
        "invalid_values": invalid_count,
    }


def save_stats(stats):
    """Save dataset statistics to JSON."""
    stats_path = PROCESSED_DIR / "dataset_stats.json"

    with stats_path.open("w", encoding="utf-8") as file:
        json.dump(stats, file, indent=2)


def main():
    print(f"Loading dataset from {RAW_MAT_PATH}...")
    images, depths = load_mat_file(RAW_MAT_PATH)

    print("Preparing array shapes...")
    images = normalize_images(images)
    depths = normalize_depths(depths)

    if len(images) != len(depths):
        raise ValueError(
            f"Images and depths count mismatch: {len(images)} images, {len(depths)} depths"
        )

    if images.shape[1:3] != depths.shape[1:3]:
        raise ValueError(
            f"Image and depth sizes do not match: {images.shape[1:3]} vs {depths.shape[1:3]}"
        )

    print("Saving images and depth maps...")
    pairs = save_dataset(images, depths)

    print("Creating train/val/test split...")
    splits, train_count, val_count, test_count = split_dataset(pairs)
    for split_name, split_pairs in splits.items():
        write_split_file(split_name, split_pairs)

    print("Saving dataset statistics...")
    stats = calculate_stats(depths, train_count, val_count, test_count)
    save_stats(stats)

    print("Done.")
    print(f"Images: {stats['num_images']}")
    print(f"Train/val/test: {train_count}/{val_count}/{test_count}")
    print(f"Invalid depth values: {stats['invalid_values']}")


if __name__ == "__main__":
    main()
