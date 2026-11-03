"""Offline correctness checks; synthetic images are not homework results."""
import csv
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from ignite.metrics import FID, InceptionScore

from data import patient_key, split_training
import evaluate as evaluation
from evaluate import InceptionEncoder
from model import VAE, vae_loss


class SmokeTests(unittest.TestCase):
    def test_forward_backward_and_loss(self):
        for size in (64, 128):
            model = VAE(image_size=size, latent_dim=8)
            x = torch.rand(2, 1, size, size)
            logits, mu, logvar = model(x)
            self.assertEqual(logits.shape, x.shape)
            loss, reconstruction, kl = vae_loss(logits, x, mu, logvar)
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.allclose(loss, reconstruction + kl))
            loss.backward()
            for module in (model.encoder, model.mu, model.logvar, model.decoder):
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()))
            with torch.no_grad():
                self.assertTrue(torch.equal(model(x, sample=False)[0], model(x, sample=False)[0]))
                images = model.generate(3)
                self.assertEqual(images.shape, (3, 1, size, size))
                self.assertTrue(((images >= 0) & (images <= 1)).all())
        x = torch.zeros(2, 1, 4, 4)
        loss, reconstruction, kl = vae_loss(x, x, torch.zeros(2, 3), torch.zeros(2, 3))
        self.assertAlmostEqual(kl.item(), 0)
        self.assertAlmostEqual(reconstruction.item(), 16 * np.log(2), places=5)

    def test_patient_grouping(self):
        paths = [Path(f"person{i}_bacteria_{j}.jpeg") for i in range(10) for j in (1, 2)]
        train, val = split_training(paths)
        self.assertFalse({patient_key(p) for p in train} & {patient_key(p) for p in val})
        self.assertEqual(len(train) + len(val), len(paths))
        self.assertEqual(patient_key("NORMAL2-IM-1427-0001.jpeg"), patient_key("NORMAL2-IM-1427-0002.jpeg"))

    def test_inception_and_metrics_without_download(self):
        with torch.inference_mode():
            # Random weights check only tensor shapes; never report these scores.
            encoder = InceptionEncoder(weights=None)
            features, probabilities = encoder(torch.rand(1, 1, 64, 64))
        self.assertEqual(features.shape, (1, 2048))
        self.assertEqual(probabilities.shape, (1, 1000))
        self.assertTrue(torch.allclose(probabilities.sum(1), torch.ones(1)))
        fid = FID(num_features=4)
        features = torch.randn(16, 4)
        fid.update((features, features.clone()))
        self.assertAlmostEqual(fid.compute(), 0, places=5)
        score = InceptionScore(num_features=3)
        score.update(torch.full((16, 3), 1 / 3))
        self.assertAlmostEqual(score.compute(), 1, places=5)

    def test_shared_resolution_validation_evaluation(self):
        # A lightweight backbone checks that the full CLI feeds BOTH sets at 64x64.
        # Real Ignite metrics still execute, with reduced dimensions for speed.
        class SmallInception(torch.nn.Module):
            def forward(self, x):
                if x.shape[1:] != (1, 64, 64):
                    raise AssertionError(f"Metric input has wrong shape: {x.shape}")
                features = torch.nn.functional.adaptive_avg_pool2d(x, (2, 2)).flatten(1)
                return features, features[:, :3].softmax(1)

        with tempfile.TemporaryDirectory(prefix="hw1-eval-smoke-") as temporary:
            root = Path(temporary)
            images = root / "data" / "train" / "NORMAL"
            images.mkdir(parents=True)
            rng = np.random.default_rng(17)
            relative_paths = []
            for i in range(6):
                path = images / f"IM-{i:04d}-0001.jpeg"
                Image.fromarray(rng.integers(0, 256, (70, 90), dtype=np.uint8)).save(path)
                relative_paths.append(str(path.relative_to(root / "data")))
            run = root / "run"
            run.mkdir()
            (run / "split.json").write_text(json.dumps({"val": relative_paths}))
            model = VAE(image_size=128, latent_dim=8)
            checkpoint = run / "best.pt"
            torch.save({"model": model.state_dict(), "config": {"image_size": 128, "latent_dim": 8},
                        "epoch": 50}, checkpoint)
            output = root / "evaluation"
            argv = ["evaluate.py", "--data", str(root / "data"), "--checkpoint", str(checkpoint),
                    "--split", "val", "--eval-image-size", "64", "--batch-size", "4",
                    "--device", "cpu", "--out", str(output)]
            with patch.object(evaluation, "InceptionEncoder", SmallInception), \
                 patch.object(evaluation, "FID", lambda **kwargs: FID(num_features=4)), \
                 patch.object(evaluation, "InceptionScore", lambda **kwargs: InceptionScore(num_features=3)), \
                 patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                evaluation.main()
            result = json.loads((output / "metrics.json").read_text())
            self.assertEqual(result["evaluation_image_size"], 64)
            self.assertEqual(result["model_image_size"], 128)
            self.assertEqual(result["real_split"], "validation (checkpoint split.json)")
            self.assertEqual(result["num_real"], 6)
            self.assertEqual(result["num_generated"], 6)
            self.assertEqual(result["latent_seed"], 124)
            self.assertTrue(np.isfinite(result["fid"]))
            self.assertTrue(np.isfinite(result["inception_score"]))
            selected = json.loads((output / "val_files.json").read_text())
            self.assertEqual(set(selected), set(relative_paths))
            self.assertFalse((output / "test_files.json").exists())
            with Image.open(output / "generated.png") as preview:
                self.assertEqual(preview.height, 128 + 4)

    def test_training_cli(self):
        with tempfile.TemporaryDirectory(prefix="hw1-smoke-") as temporary:
            root = Path(temporary)
            folder = root / "chest_xray" / "train" / "NORMAL"
            folder.mkdir(parents=True)
            rng = np.random.default_rng(42)
            for i in range(12):
                image = rng.integers(0, 256, (70, 90), dtype=np.uint8)
                Image.fromarray(image).save(folder / f"IM-{i:04d}-0001.jpeg")
            output = root / "run"
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = "2"
            subprocess.run([sys.executable, str(Path(__file__).with_name("train.py")),
                            "--data", str(root), "--out", str(output), "--epochs", "1",
                            "--batch-size", "4", "--latent-dim", "8", "--device", "cpu"],
                           check=True, env=env, timeout=120)
            for name in ("best.pt", "last.pt", "history.csv", "loss_curves.png", "reconstructions.png",
                         "best_samples.png", "split.json", "config.json"):
                self.assertTrue((output / name).is_file(), name)
            with (output / "history.csv").open() as stream:
                row = next(csv.DictReader(stream))
            self.assertTrue(np.isfinite(float(row["train_loss"])))
            config = json.loads((output / "config.json").read_text())
            self.assertEqual(config["n_train"] + config["n_val"], 12)
            checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
            restored = VAE(config["image_size"], config["latent_dim"])
            restored.load_state_dict(checkpoint["model"])


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
