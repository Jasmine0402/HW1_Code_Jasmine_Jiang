"""Train an unconditional VAE and save checkpoints, curves and image grids."""
import argparse
import csv
import json
import platform
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from data import XRayDataset, find_root, image_paths, split_training
from model import VAE, select_device, vae_loss


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, device, beta, optimizer=None):
    training = optimizer is not None
    model.train(training)
    totals = np.zeros(3)
    count = 0
    with torch.set_grad_enabled(training):
        for x in loader:
            x = x.to(device)
            # Stochastic validation estimates the same VAE objective as training.
            logits, mu, logvar = model(x)
            loss, reconstruction, kl = vae_loss(logits, x, mu, logvar, beta)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss; check input data and learning rate")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            totals += np.array([loss.item(), reconstruction.item(), kl.item()]) * len(x)
            count += len(x)
    return dict(zip(("loss", "reconstruction", "kl"), (totals / count).tolist()))


def plot_history(history, path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, metric in zip(axes, ("loss", "reconstruction", "kl")):
        for split in ("train", "val"):
            ax.plot([r["epoch"] for r in history], [r[f"{split}_{metric}"] for r in history], label=split)
        ax.set(xlabel="Epoch", ylabel="Loss per image", title=metric)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default="runs/baseline")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--image-size", type=int, choices=(64, 128), default=64)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--max-train", type=int, default=None, help="Subset for smoke runs only")
    parser.add_argument("--max-val", type=int, default=None, help="Subset for smoke runs only")
    args = parser.parse_args()
    if (args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.beta < 0
            or not 0 < args.val_fraction < 1 or args.workers < 0
            or any(v is not None and v < 1 for v in (args.max_train, args.max_val))):
        parser.error("Invalid positive count, learning rate, beta, or validation fraction")
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        parser.error(f"Output directory is not empty: {out}. Choose a new --out.")
    seed_all(args.seed)
    device = select_device(args.device)
    root = find_root(args.data)
    train_paths, val_paths = split_training(image_paths(root / "train"), args.val_fraction, args.seed)
    if args.max_train:
        train_paths = train_paths[:args.max_train]
    if args.max_val:
        val_paths = val_paths[:args.max_val]
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "data_root": str(root),
        "split_method": "filename-inferred patient groups from official train; official val unused",
        "train": [str(p.relative_to(root)) for p in train_paths],
        "val": [str(p.relative_to(root)) for p in val_paths],
    }
    (out / "split.json").write_text(json.dumps(manifest, indent=2))
    config = vars(args) | {"device_used": str(device), "n_train": len(train_paths), "n_val": len(val_paths),
                           "torch_version": str(torch.__version__), "torchvision_version": str(torchvision.__version__),
                           "python_version": platform.python_version()}
    (out / "config.json").write_text(json.dumps(config, indent=2))
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(XRayDataset(train_paths, args.image_size), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers, generator=generator)
    val_loader = DataLoader(XRayDataset(val_paths, args.image_size), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers)
    model = VAE(args.image_size, args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Generate on CPU first for portability across CUDA/MPS/CPU random generators.
    fixed_z = torch.randn(64, args.latent_dim, generator=torch.Generator().manual_seed(args.seed + 1)).to(device)
    fixed_x = next(iter(val_loader))[:8].to(device)
    save_image(fixed_x.cpu(), out / "real_examples.png", nrow=8)
    history = []
    best_loss = float("inf")
    print(f"Device: {device}; train: {len(train_paths)}; val: {len(val_paths)}", flush=True)
    print(f"Config: image={args.image_size}x{args.image_size}; latent={args.latent_dim}; "
          f"batch={args.batch_size}; epochs={args.epochs}; lr={args.lr:g}; beta={args.beta:g}", flush=True)
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        train = run_epoch(model, train_loader, device, args.beta, optimizer)
        val = run_epoch(model, val_loader, device, args.beta)
        row = {"epoch": epoch, "seconds": time.perf_counter() - start,
               **{f"train_{k}": v for k, v in train.items()}, **{f"val_{k}": v for k, v in val.items()}}
        history.append(row)
        with (out / "history.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        plot_history(history, out / "loss_curves.png")
        model.eval()
        with torch.no_grad():
            save_image(model.generate(64, fixed_z).cpu(), out / f"samples_epoch_{epoch:03d}.png", nrow=8)
            logits, _, _ = model(fixed_x, sample=False)
            save_image(torch.cat([fixed_x, logits.sigmoid()]).cpu(), out / "reconstructions.png", nrow=len(fixed_x))
        checkpoint = {"model": model.state_dict(), "config": config, "epoch": epoch, "val_loss": val["loss"]}
        torch.save(checkpoint, out / "last.pt")
        if val["loss"] < best_loss:
            best_loss = val["loss"]
            torch.save(checkpoint, out / "best.pt")
            with torch.no_grad():
                save_image(model.generate(64, fixed_z).cpu(), out / "best_samples.png", nrow=8)
        print(f"Epoch {epoch:03d} | train {train['loss']:.2f} | val {val['loss']:.2f} "
              f"| recon {val['reconstruction']:.2f} | KL {val['kl']:.2f} | {row['seconds']:.1f}s", flush=True)
    print(f"Saved results to {out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
