"""Evaluate fresh prior samples on test or checkpoint validation images."""
import argparse
import json
from pathlib import Path

import ignite
import torch
import torchvision
from ignite.metrics import FID, InceptionScore
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.models import Inception_V3_Weights, inception_v3
from torchvision.utils import save_image

from data import XRayDataset, find_root, image_paths
from model import VAE, select_device


class InceptionEncoder(nn.Module):
    def __init__(self, weights=Inception_V3_Weights.DEFAULT):
        super().__init__()
        # weights=None is used ONLY in the offline smoke test, never in main().
        self.backbone = inception_v3(weights=weights, init_weights=False, transform_input=True)
        self.classifier = self.backbone.fc
        self.backbone.fc = nn.Identity()
        self.preprocess = Inception_V3_Weights.DEFAULT.transforms(antialias=True)
        self.eval()
        self.requires_grad_(False)

    def forward(self, x):
        # Repeat grayscale to RGB and use the official pretrained preprocessing.
        x = x.expand(-1, 3, -1, -1)
        features = self.backbone(self.preprocess(x))
        probabilities = self.classifier(features).softmax(dim=1)
        return features, probabilities


def resize_for_metrics(images, image_size):
    """Put native model outputs on a shared resolution before Inception."""
    if images.shape[-2:] == (image_size, image_size):
        return images
    return F.interpolate(images, size=(image_size, image_size), mode="bilinear",
                         align_corners=False, antialias=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", default="runs/baseline/best.pt")
    parser.add_argument("--out", default=None, help="Default: checkpoint directory / evaluation")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-samples", type=int, default=None, help="Default: all images in the selected split")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--split", choices=("test", "val"), default="test",
                        help="Use checkpoint split.json for validation during tuning")
    parser.add_argument("--eval-image-size", type=int, choices=(64, 128), default=None,
                        help="Shared metric resolution; default: native checkpoint resolution")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    device = select_device(args.device)
    root = find_root(args.data)
    if args.split == "val":
        split_path = Path(args.checkpoint).parent / "split.json"
        if not split_path.is_file():
            parser.error(f"Validation evaluation requires {split_path}")
        split = json.loads(split_path.read_text())
        if not split.get("val"):
            parser.error(f"No validation images listed in {split_path}")
        paths = [root / relative for relative in split["val"]]
        missing = [p for p in paths if not p.is_file()]
        if missing:
            parser.error(f"Missing validation image: {missing[0]}")
        real_split = "validation (checkpoint split.json)"
    else:
        paths = image_paths(root / "test")
        real_split = "official test"
    n = len(paths) if args.num_samples is None else args.num_samples
    if not 2 <= n <= len(paths):
        parser.error(f"--num-samples must be between 2 and {len(paths)}")
    # Select the subset without ordering NORMAL before PNEUMONIA.
    selection_rng = torch.Generator().manual_seed(args.seed)
    paths = [paths[i] for i in torch.randperm(len(paths), generator=selection_rng)[:n].tolist()]
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    model = VAE(config["image_size"], config["latent_dim"])
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    evaluation_size = args.eval_image_size or config["image_size"]
    loader = DataLoader(XRayDataset(paths, evaluation_size), batch_size=args.batch_size)
    if args.out:
        out = Path(args.out)
    elif args.split == "test" and args.eval_image_size is None:
        out = Path(args.checkpoint).parent / "evaluation"
    else:
        out = Path(args.checkpoint).parent / f"evaluation_{args.split}_{evaluation_size}"
    if out.exists() and any(out.iterdir()):
        parser.error(f"Output directory is not empty: {out}. Choose a new --out.")
    out.mkdir(parents=True, exist_ok=True)
    print(f"Evaluating {n} real and {n} generated images on {device}. "
          "First run downloads pretrained Inception weights (~104 MB).", flush=True)
    print(f"Reference: {real_split}; model size: {config['image_size']}; "
          f"evaluation size: {evaluation_size}; latent seed: {args.seed + 1}", flush=True)
    inception = InceptionEncoder().to(device)
    # Accumulate on CPU: these metrics use float64, which MPS does not support.
    fid = FID(num_features=2048, feature_extractor=nn.Identity(), device="cpu")
    score = InceptionScore(num_features=1000, feature_extractor=nn.Identity(), device="cpu")
    sample_rng = torch.Generator().manual_seed(args.seed + 1)
    preview = []
    completed = 0
    with torch.inference_mode():
        for real in loader:
            z = torch.randn(len(real), config["latent_dim"], generator=sample_rng).to(device)
            fake = model.generate(len(real), z=z)
            fake_for_metrics = resize_for_metrics(fake, evaluation_size)
            fake_features, probabilities = inception(fake_for_metrics)
            real_features, _ = inception(real.to(device))
            fid.update((fake_features.cpu(), real_features.cpu()))
            score.update(probabilities.cpu())
            if completed < 64:
                preview.append(fake[:64 - completed].cpu())
            completed += len(real)
            print(f"Features: {completed}/{n}", flush=True)
    print("Computing FID covariance matrix square root on CPU...", flush=True)
    results = {
        "fid": float(fid.compute()), "inception_score": float(score.compute()),
        "num_real": n, "num_generated": n, "seed": args.seed, "latent_seed": args.seed + 1,
        "checkpoint": str(Path(args.checkpoint).resolve()), "checkpoint_epoch": checkpoint["epoch"],
        "real_split": real_split, "generation": "independent z ~ N(0, I), not reconstructions",
        "model_image_size": config["image_size"], "evaluation_image_size": evaluation_size,
        "preview_image_size": config["image_size"],
        "fid_features": "2048-d pooled torchvision Inception-v3, ImageNet weights",
        "preprocessing": (
            f"Real: PIL grayscale resize to {evaluation_size}x{evaluation_size}. "
            f"Generated: native {config['image_size']}x{config['image_size']}, then tensor bilinear "
            f"antialiased resize to {evaluation_size}x{evaluation_size} if needed. "
            "Both: repeat grayscale to RGB; official Inception weights.transforms with antialias=True."
        ),
        "score_protocol": "Ignite single-set IS; no split mean/std",
        "torch_version": torch.__version__, "torchvision_version": torchvision.__version__,
        "ignite_version": ignite.__version__,
        "limitations": [
            "ImageNet features/classes are not chest-X-ray-specific or a clinical validity measure.",
            "Finite-sample FID is sample-size dependent; report sample counts with scores.",
            "Torchvision features/preprocessing differ from pytorch-fid; scores are not directly interchangeable.",
        ],
    }
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    manifest_name = "val_files.json" if args.split == "val" else "test_files.json"
    (out / manifest_name).write_text(json.dumps([str(p.relative_to(root)) for p in paths], indent=2))
    save_image(torch.cat(preview), out / "generated.png", nrow=8)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
