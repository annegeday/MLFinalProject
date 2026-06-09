"""
FSD50K — Training Setup
========================
This file is the shared foundation for all model experiments.
It covers:
  1. Imports and config
  2. Label encoder
  3. Dataset class  (loads .pt files, pads/truncates, normalizes)
  4. DataLoaders
  5. Sanity check

Copy this file into any model experiment and add your model below.

Folder layout expected:
  preprocessed/
  ├── train/   ← .pt files from preprocessing script
  ├── val/
  └── eval/
  data/
  └── FSD50K.ground_truth/
      ├── dev.csv
      ├── eval.csv
      └── vocabulary.csv
"""

# ─────────────────────────────────────────────
# 1. IMPORTS AND CONFIG
# ─────────────────────────────────────────────
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Subset
import math
import random
random.seed(42)

subset = True

# Paths
GT_DEV       = Path("data/FSD50K.ground_truth/dev.csv")
GT_EVAL      = Path("data/FSD50K.ground_truth/eval.csv")
GT_VOCAB     = Path("data/FSD50K.ground_truth/vocabulary.csv")
PREPROCESSED = Path("preprocessed")

# Normalization constants — Insert from save specs notebook 
NORM_MEAN = -17.9358
NORM_STD  =  20.7367

# Clip length - can be changed to test different values
CLIP_SECONDS = 6    # seconds

# Preprocessing parameters, these must match the values used in save specs notebook
SAMPLE_RATE  = 22050                      
HOP_LENGTH   = 256                        

# Compute frame count
FRAME_COUNT  = int(CLIP_SECONDS * SAMPLE_RATE / HOP_LENGTH)  # time frames

# Training - batch size can be changed to test different values
BATCH_SIZE   = 8
NUM_WORKERS  = 0     # change to match needed/available worker


# ─────────────────────────────────────────────
# 2. LABEL ENCODER
# ─────────────────────────────────────────────
# Label encoder - maps between label names and indices

vocab = pd.read_csv(GT_VOCAB, header=None, names=["index", "label_name", "mid"])

label_to_idx = {row["label_name"]: row["index"] for _, row in vocab.iterrows()}
idx_to_label = {v: k for k, v in label_to_idx.items()}
NUM_CLASSES  = len(label_to_idx)   # 200

def encode_labels(label_string: str) -> torch.Tensor:
    """'Electric_guitar,Guitar,Music' → float tensor of shape (200,)"""
    vec = torch.zeros(NUM_CLASSES, dtype=torch.float32)
    for lbl in label_string.split(","):
        lbl = lbl.strip()
        if lbl in label_to_idx:
            vec[label_to_idx[lbl]] = 1.0
    return vec

def decode_labels(label_vec: torch.Tensor) -> list:
    """float tensor of shape (200,) → list of active class name strings"""
    return [idx_to_label[i] for i in label_vec.nonzero().flatten().tolist()]


# ─────────────────────────────────────────────
# 3. DATASET CLASS
# ─────────────────────────────────────────────
# Customized Torch Dataset class: loads preprocessed spectrograms, pads/truncates, normalizes, and returns (spec, label) pairs
# Essentially this loads the data in a way that torch can use for training, and applies the necessary transformations when needed to avoid doing them all at once in memory

class FSD50KDataset(Dataset):
    """
    Loads preprocessed .pt spectrograms, pads or truncates to FRAME_COUNT,
    normalizes, and returns (spectrogram, label) pairs.

    Parameters
    ----------
    df        : DataFrame with columns [fname, labels]
    split     : "train", "val", or "eval" — subfolder inside preprocessed/
    """

    def __init__(self, df: pd.DataFrame, split: str):
        self.split_dir = PREPROCESSED / split
        # Pre-build lists for fast access — avoids DataFrame lookup on every sample
        self.fnames = df["fname"].tolist()
        self.labels = df["labels"].tolist()

    def __len__(self) -> int:
        return len(self.fnames)

    def __getitem__(self, idx: int):
        path = self.split_dir / f"{self.fnames[idx]}.pt"

        # 1. Load preprocessed spectrogram — shape (1, 128, time_frames)
        spec = torch.load(path, weights_only=True)

        # 2. Repeat or truncate to FRAME_COUNT
        n = spec.shape[-1]
        if n < FRAME_COUNT:
            repeats = math.ceil(FRAME_COUNT / n)
            spec = spec.repeat(1, 1, repeats)
        spec = spec[:, :, :FRAME_COUNT]

        # 3. Normalize
        spec = (spec - NORM_MEAN) / (NORM_STD + 1e-6)

        # 4. Encode labels
        label = encode_labels(self.labels[idx])

        return spec, label
# ─────────────────────────────────────────────
# 4. DATALOADERS
# ─────────────────────────────────────────────
# Dataloaders for training, validation, and evaluation. This is where the datasets are actually loaded and prepared for training.
# The DataLoader handles batching, shuffling, and parallel loading.

dev_csv  = pd.read_csv(GT_DEV)
eval_csv = pd.read_csv(GT_EVAL)

train_df = dev_csv[dev_csv["split"] == "train"]
val_df   = dev_csv[dev_csv["split"] == "val"]
eval_df  = eval_csv

train_dataset = FSD50KDataset(train_df, "train")
if subset == True:
    # Use a random subset of 2000 clips for fast iteration
    subset_indices = random.sample(range(len(train_dataset)), 2000)
    train_dataset  = Subset(train_dataset, subset_indices)

val_dataset   = FSD50KDataset(val_df,   "val")
eval_dataset  = FSD50KDataset(eval_df,  "eval")

train_loader = DataLoader(
    train_dataset,
    batch_size  = BATCH_SIZE,
    shuffle     = True,
    num_workers = NUM_WORKERS,
    pin_memory  = True,             # speeds up transfer to GPU if available
)
val_loader = DataLoader(
    val_dataset,
    batch_size  = BATCH_SIZE,
    shuffle     = False,
    num_workers = NUM_WORKERS,
    pin_memory  = True,
)
eval_loader = DataLoader(
    eval_dataset,
    batch_size  = BATCH_SIZE,
    shuffle     = False,
    num_workers = NUM_WORKERS,
    pin_memory  = True,
)


# ─────────────────────────────────────────────
# 5. SANITY CHECK
# ─────────────────────────────────────────────
# Sanity check: load one batch and print shapes and label decoding to verify everything works as expected

if __name__ == "__main__":
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Eval: {len(eval_dataset)}")
    print(f"Clip length: {CLIP_SECONDS}s → {FRAME_COUNT} time frames")
    print(f"Spectrogram shape per clip: (1, 128, {FRAME_COUNT})")

    features, labels = next(iter(train_loader))
    print(f"\nFeature batch shape : {features.shape}")   # (32, 1, 128, FRAME_COUNT)
    print(f"Label batch shape   : {labels.shape}")       # (32, 200)
    print(f"\nFirst clip labels: {decode_labels(labels[0])}")

# means, stds = [], []
# for i, (features, _) in enumerate(train_loader):
#     means.append(features.mean().item())
#     stds.append(features.std().item())
#     if i == 20:   # 20 batches is enough
#         break

# print(f"Mean of batch means : {sum(means)/len(means):.4f}")  # should be ~0
# print(f"Mean of batch stds  : {sum(stds)/len(stds):.4f}")    # should be ~1
