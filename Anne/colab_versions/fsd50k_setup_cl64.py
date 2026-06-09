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
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from torch.utils.data import WeightedRandomSampler

warnings.filterwarnings("ignore", category=UndefinedMetricWarning) # for metrics that may be undefined on small subsets
random.seed(42)

subset = False
oversample = True
specaug = False
focalloss = False

base = "/content/drive/MyDrive/ML_Final_Project"

# Paths
GT_DEV       = Path(f"{base}/data/FSD50K.ground_truth/dev.csv")
GT_EVAL      = Path(f"{base}/data/FSD50K.ground_truth/eval.csv")
GT_VOCAB     = Path(f"{base}/data/FSD50K.ground_truth/vocabulary.csv")
PREPROCESSED = Path("/content/preprocessed")

# Normalization constants — Insert from save specs notebook 
NORM_MEAN = -21.5820
NORM_STD  =  20.2802

# Clip length - can be changed to test different values
CLIP_SECONDS = 6    # seconds

# Preprocessing parameters, these must match the values used in save specs notebook
SAMPLE_RATE  = 22050                      
HOP_LENGTH   = 256                        

# Compute frame count
FRAME_COUNT  = int(CLIP_SECONDS * SAMPLE_RATE / HOP_LENGTH)  # time frames

# Training - batch size can be changed to test different values
BATCH_SIZE   = 32
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

def spec_augment(spec, freq_masks=2, time_masks=2, freq_width=8, time_width=40):
    """Apply SpecAugment: random frequency and time masking."""
    spec = spec.clone()
    _, n_mels, n_frames = spec.shape

    for _ in range(freq_masks):
        f = random.randint(0, freq_width)
        f0 = random.randint(0, n_mels - f)
        spec[:, f0:f0+f, :] = 0.0

    for _ in range(time_masks):
        t = random.randint(0, time_width)
        t0 = random.randint(0, n_frames - t)
        spec[:, :, t0:t0+t] = 0.0

    return spec

class FSD50KDataset(Dataset):
    """
    Loads preprocessed .pt spectrograms, pads or truncates to FRAME_COUNT,
    normalizes, and returns (spectrogram, label) pairs.

    Parameters
    ----------
    df        : DataFrame with columns [fname, labels]
    split     : "train", "val", or "eval" — subfolder inside preprocessed/
    """

    def __init__(self, df: pd.DataFrame, split: str, spec_augmentation: bool = False):
        self.split_dir = PREPROCESSED / split
        # Pre-build lists for fast access — avoids DataFrame lookup on every sample
        self.fnames = df["fname"].tolist()
        self.labels = df["labels"].tolist()
        self.training = spec_augmentation

    def __len__(self) -> int:
        return len(self.fnames)

    def __getitem__(self, idx: int):
        path = self.split_dir / f"{self.fnames[idx]}.pt"

        # 1. Load preprocessed spectrogram — shape (1, 128, time_frames)
        spec = torch.load(path, weights_only=True)

        # 2. Pad if too short, then crop to FRAME_COUNT
        n = spec.shape[-1]
        if n < FRAME_COUNT:
            repeats = math.ceil(FRAME_COUNT / n)
            spec = spec.repeat(1, 1, repeats)

        # Random crop
        start = random.randint(0, spec.shape[-1] - FRAME_COUNT)
        spec  = spec[:, :, start:start + FRAME_COUNT]

        # 3. Normalize
        spec = (spec - NORM_MEAN) / (NORM_STD + 1e-6)

        if self.training:                     
            spec = spec_augment(spec)

        # 4. Encode labels
        label = encode_labels(self.labels[idx])

        return spec, label
    
def make_weighted_sampler(dataset, df):
    """Compute per-clip sampling weights based on rarest active label."""
    # Class frequencies
    all_labels = torch.stack([encode_labels(lbl) for lbl in df["labels"]])
    class_counts = all_labels.sum(dim=0).clamp(min=1)
    class_weights = 1.0 / class_counts  # rare classes get higher weight

    # Per-clip weight = weight of rarest active label
    clip_weights = []
    for lbl_str in df["labels"]:
        vec = encode_labels(lbl_str)
        active = vec.nonzero().flatten()
        if len(active) == 0:
            clip_weights.append(1.0)
        else:
            clip_weights.append(class_weights[active].max().item())

    return WeightedRandomSampler(
        weights     = clip_weights,
        num_samples = len(clip_weights),
        replacement = True
    )

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

train_dataset = FSD50KDataset(train_df, "train", spec_augmentation=specaug)
val_dataset   = FSD50KDataset(val_df,   "val")
eval_dataset  = FSD50KDataset(eval_df,  "eval")

if subset:
    subset_indices = random.sample(range(len(train_dataset)), 2000)
    train_dataset  = Subset(train_dataset, subset_indices)
    val_indices    = random.sample(range(len(val_dataset)), 500)
    val_dataset    = Subset(val_dataset, val_indices)


if oversample and not subset:
    sampler = make_weighted_sampler(train_dataset, train_df)
    train_loader = DataLoader(
        train_dataset,
        batch_size  = BATCH_SIZE,
        sampler     = sampler,        # sampler replaces shuffle
        num_workers = NUM_WORKERS,
        pin_memory  = True,
    )
elif oversample and subset:
    # Milder oversampling on subset — just use average weight instead of rarest
    all_labels_sub = torch.stack([encode_labels(lbl) for lbl in train_df["labels"].iloc[subset_indices]])
    class_counts   = all_labels_sub.sum(dim=0).clamp(min=1)
    class_weights  = 1.0 / class_counts

    clip_weights = []
    for lbl_str in train_df["labels"].iloc[subset_indices]:
        vec    = encode_labels(lbl_str)
        active = vec.nonzero().flatten()
        if len(active) == 0:
            clip_weights.append(1.0)
        else:
            clip_weights.append(class_weights[active].mean().item())  # average, not max

    sampler = WeightedRandomSampler(clip_weights, len(clip_weights), replacement=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size  = BATCH_SIZE,
        sampler     = sampler,
        num_workers = NUM_WORKERS,
        pin_memory  = True,
    )
else:
    train_loader = DataLoader(
        train_dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = True,
        num_workers = NUM_WORKERS,
        pin_memory  = True,
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
    print(f"\nFeature batch shape : {features.shape}")   # (32, 1, n_mel, FRAME_COUNT)
    print(f"Label batch shape   : {labels.shape}")       # (32, 200)
    print(f"\nFirst clip labels: {decode_labels(labels[0])}")

# ─────────────────────────────────────────────
# 6. MAP CLASSES TO FAMILIES
# ─────────────────────────────────────────────
import urllib.request
import json

url = "https://raw.githubusercontent.com/audioset/ontology/master/ontology.json"

if not Path("ontology.json").exists():
    urllib.request.urlretrieve(url, "ontology.json")

with open("ontology.json") as f:
    ontology = json.load(f)

ontology_by_id = {entry["id"]: entry for entry in ontology}

child_to_parent = {}
for entry in ontology:
    for child_id in entry["child_ids"]:
        child_to_parent[child_id] = entry["id"]

top_level = {
    "/m/0dgw9r": "Human sounds",
    "/m/0jbk":   "Animal",
    "/m/04rlf":  "Music",
    "/m/059j3w": "Natural sounds",
    "/t/dd00041": "Sounds of things",
    "/t/dd00098": "Source-ambiguous sounds",
}

def get_top_level_family(mid):
    current = mid
    while current not in top_level:
        parent = child_to_parent.get(current)
        if parent is None:
            return "Unknown"
        current = parent
    return top_level[current]

vocab["family"] = vocab["mid"].apply(get_top_level_family)

def get_clip_family(label_string):
    for lbl in label_string.split(","):
        vocab_row = vocab[vocab["label_name"] == lbl]
        if not vocab_row.empty:
            return vocab_row.iloc[0]["family"]
    return "Unknown"