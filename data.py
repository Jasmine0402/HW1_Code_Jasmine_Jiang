"""Read Kaggle's images; keep filename-inferred patients together in splits."""
import random
import re
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

EXTENSIONS = {".jpeg", ".jpg", ".png"}


def find_root(path):
    path = Path(path).expanduser().resolve()
    for candidate in (path, path / "chest_xray", path / "chest_xray" / "chest_xray"):
        if (candidate / "train").is_dir():
            return candidate
    raise FileNotFoundError(f"No train/ directory under {path}. Set --data to the extracted dataset.")


def image_paths(folder):
    paths = sorted(p for p in Path(folder).rglob("*")
                   if p.suffix.lower() in EXTENSIONS and not p.name.startswith("."))
    if not paths:
        raise FileNotFoundError(f"No images in {folder}")
    return paths


def patient_key(path):
    # This is an inference from filenames, not verified clinical patient metadata.
    stem = Path(path).stem
    match = re.search(r"person\d+|(?:NORMAL2-)?IM-\d+", stem, re.I)
    return match.group(0).lower() if match else stem


def split_training(paths, val_fraction=0.1, seed=42):
    groups = {}
    for path in paths:
        groups.setdefault(patient_key(path), []).append(path)
    keys = sorted(groups)
    if len(keys) < 2:
        raise ValueError("At least two patient groups are required")
    random.Random(seed).shuffle(keys)
    n_val = max(1, min(len(keys) - 1, round(len(keys) * val_fraction)))
    val_keys = set(keys[:n_val])
    train = [p for k in keys if k not in val_keys for p in groups[k]]
    val = [p for k in keys if k in val_keys for p in groups[k]]
    return train, val


class XRayDataset(Dataset):
    def __init__(self, paths, image_size=64):
        self.paths = list(paths)
        # Resize directly: preserves the complete image but changes aspect ratio.
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("L"))
