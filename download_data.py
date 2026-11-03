from pathlib import Path

import kagglehub
from data import find_root

save_dir = Path(__file__).resolve().parent / "data"

path = kagglehub.dataset_download(
    "paultimothymooney/chest-xray-pneumonia",
    output_dir=str(save_dir),
)

data_root = find_root(path)

print("\nDataset Download Finished")
print("Data Root：", data_root)
print(
    f'python train.py --data "{data_root}" '
    "--out runs/quick --epochs 1 --max-train 256 --max-val 64"
)