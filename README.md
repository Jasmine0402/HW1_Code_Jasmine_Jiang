# ECE 285 Homework 1 — Jasmine Jiang

This code implements the original convolutional VAE baseline described in the report.
Submission defaults: 64x64 grayscale images, 64 latent dimensions, beta=1,
Adam lr=0.001, batch size 64, 30 epochs, and training seed 42.
The default output directory is `runs/baseline`.

## Files

- `model.py`: encoder, reparameterization, decoder, VAE objective, device selection.
- `data.py`: image loading and filename-inferred patient-group train/validation split.
- `train.py`: training, validation, checkpoint selection, image grids and loss plots.
- `evaluate.py`: independent prior sampling and Ignite FID/Inception Score evaluation.
- `download_data.py`: download the assigned Kaggle dataset using KaggleHub.
- `smoke_test.py`: offline checks with synthetic data, including checkpoint loading and evaluation plumbing.
- `requirements.txt`: versions installed in the environment used for the experiments.

The archive contains source code and instructions. The dataset, virtual environment,
trained weights and generated run outputs are not bundled.

## Setup

Use Python 3.13 (the experiment used Python 3.13.9). From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python smoke_test.py
```

The offline checks use synthetic images and do not produce homework experiment results.
They do not require the chest-X-ray dataset or a pretrained-model download.

## Download the dataset

```bash
python download_data.py
```

KaggleHub downloads and extracts the assigned dataset under `./data`.
The script prints the resolved image directory. With the current dataset layout it is
`./data/chest_xray`; use the printed path if the layout differs.
If Kaggle requires authentication or consent, complete its prompts separately.
Dataset: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

## Train the ORIGINAL baseline

```bash
python train.py --data ./data/chest_xray --out runs/baseline --image-size 64 --epochs 30 --lr 0.001 --batch-size 64 --latent-dim 64 --beta 1 --seed 42
```

The script selects CUDA, Apple MPS or CPU automatically. To force CPU, append `--device cpu`.
An output directory must be empty; choose another `--out` for a new run.
The original run used 4,649 training and 567 validation images. The official test set was
not used for parameter updates. The checkpoint with the lowest validation total loss is
saved as `best.pt`; the original run selected epoch 29.

Output includes `config.json`, `split.json`, `history.csv`, `loss_curves.png`,
`best.pt`, `last.pt`, `best_samples.png`, and per-epoch image grids.
Seeds aid reproducibility; exact numerical results can vary across hardware/software.

## Evaluate on the official TEST set

After training:

```bash
python evaluate.py --data ./data/chest_xray --checkpoint runs/baseline/best.pt --split test --eval-image-size 64 --out runs/baseline/evaluation
```

The first evaluation downloads ImageNet-pretrained Inception-v3 weights (~104 MB).
The script generates one fresh prior sample for each of the 624 test images.
It writes `metrics.json`, `test_files.json` and `generated.png` in the evaluation directory.
Both real/generated tensors use 64x64 grayscale data, repeated into RGB, followed by
the official Inception weights transforms (antialiased resize to 342, center crop to 299,
ImageNet normalization). FID uses 2048-dimensional pooled torchvision features; IS uses
1000-class softmax probabilities. The Ignite metric accumulators run on CPU.

Reported ORIGINAL test results: FID=228.00541383938324; IS=1.594599206441914;
624 real images / 624 generated images, checkpoint epoch 29.
The evaluation root seed is 123; independent latent sampling uses seed 124.
IS is one whole-set score, not a split mean/std. These are generative-image metrics,
not disease-classification accuracy or evidence of clinical validity.

## Optional evaluation controls

`--split val` reads validation paths from the checkpoint directory's `split.json`.
`--eval-image-size` sets a common metric resolution (64 or 128); it does not resize the model.
Without these options, the original test/native-resolution behavior is retained.

## References

- Kingma and Welling. Auto-Encoding Variational Bayes. https://arxiv.org/abs/1312.6114
- Assigned Ignite tutorial: https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/
- Ignite FID API: https://docs.pytorch.org/ignite/generated/ignite.metrics.FID.html

The evaluation adapts the tutorial's FID and InceptionScore classes to VAE sampling.
The explicit pooled features and official-weight preprocessing differ from the tutorial's
defaults and PIL resizing; the report describes the protocol used for the reported scores.
