# ============================================================================
# PHASE 2: MODEL TRAINING (v2.6 - CLASSIFICATION, IMPROVED v2.5)
#
# Goals:
# - Improve generalization + stabilize minority classes (M3/M4)
# - Use pretrained CNN backbone + temporal head (LSTM + attention)
# - Use robust augmentations and class-imbalance-aware loss
#
# Key Changes vs v2.5:
# - Task 2 becomes 4-class moment-intensity:
#   0: No Pain (M0/M1), 1: Acute (M2), 2: Declining (M3), 3: Recovery (M4)
# - Pretrained backbone (ImageNet) + AMP for L4 GPU
# - Optional: class-balanced focal loss + label smoothing
#
# References:
# - Focal Loss: Lin et al., ICCV 2017
# - Class-Balanced Loss: Cui et al., CVPR 2019
# - Label Smoothing: Szegedy et al., CVPR 2016
# - AugMix: Hendrycks et al., ICLR 2020
# - MixStyle: Zhou et al., ICLR 2021
# ============================================================================

import os
import json
import math
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm

import torchvision
import torchvision.transforms.functional as TF
from torchvision.models import ResNet18_Weights

from PIL import Image

warnings.filterwarnings("ignore")


# ----------------------------
# Reproducibility utilities
# ----------------------------
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinism is helpful for CV comparisons, but may reduce speed on GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------
# Config (Colab/L4 friendly)
# ----------------------------
@dataclass
class Config:
    # Data
    max_frames: int = 32
    resolution: Tuple[int, int] = (112, 112)
    num_workers: int = 2

    # CV
    num_folds: int = 9
    test_animals: Tuple[int, int] = (14, 17)

    # Training
    num_epochs: int = 50
    batch_size: int = 16  # L4 safe; tune upward if memory allows
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    warmup_epochs: int = 2

    # Temporal head
    lstm_hidden_size: int = 256
    dropout: float = 0.3

    # Loss/imbalance
    use_weighted_sampler: bool = True
    use_class_balanced: bool = True  # Cui et al.
    cb_beta: float = 0.9999
    use_focal: bool = True  # Lin et al.
    focal_gamma: float = 2.0
    label_smoothing: float = 0.05  # Szegedy et al. (for CE head)

    # Augmentation
    aug_prob: float = 0.9
    use_random_erasing: bool = True
    random_erasing_prob: float = 0.25

    # Paths (Colab default: /content/drive/MyDrive)
    project_dirname: str = "facial_pain_project_v2"
    checkpoint_subdir: str = "checkpoints_v2.6"
    results_subdir: str = "results_v2.6"


# ----------------------------
# Label mapping for v2.6
# ----------------------------
def moment_to_task1_binary(moment: str) -> int:
    # Pain vs No Pain (keep same as v2.5)
    return 1 if moment in ["M2", "M3", "M4"] else 0


def moment_to_task2_4class(moment: str) -> int:
    # 0: No Pain (M0/M1), 1: Acute (M2), 2: Declining (M3), 3: Recovery (M4)
    if moment in ["M0", "M1"]:
        return 0
    if moment == "M2":
        return 1
    if moment == "M3":
        return 2
    if moment == "M4":
        return 3
    return 0


# ----------------------------
# Video augmentation helpers
# (apply same random params across frames)
# ----------------------------
class VideoAugment:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def __call__(self, frames: List[Image.Image]) -> List[Image.Image]:
        if random.random() > self.cfg.aug_prob:
            return frames

        # Horizontal flip
        do_hflip = random.random() < 0.5

        # Color jitter params (consistent across frames)
        brightness = 1.0 + random.uniform(-0.25, 0.25)
        contrast = 1.0 + random.uniform(-0.25, 0.25)
        saturation = 1.0 + random.uniform(-0.25, 0.25)

        # Small rotation
        angle = random.uniform(-8, 8)

        out = []
        for img in frames:
            if do_hflip:
                img = TF.hflip(img)
            img = TF.adjust_brightness(img, brightness)
            img = TF.adjust_contrast(img, contrast)
            img = TF.adjust_saturation(img, saturation)
            img = TF.rotate(img, angle, fill=0)
            out.append(img)
        return out


class VideoToTensorNormalize:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        weights = ResNet18_Weights.DEFAULT
        self.mean = weights.transforms().mean
        self.std = weights.transforms().std

    def __call__(self, frames: List[Image.Image]) -> torch.Tensor:
        tensors = []
        for img in frames:
            img = TF.resize(img, self.cfg.resolution)
            t = TF.to_tensor(img)
            t = TF.normalize(t, mean=self.mean, std=self.std)
            tensors.append(t)
        return torch.stack(tensors, dim=0)  # (T, C, H, W)


# ----------------------------
# Dataset
# ----------------------------
class FacialPainDataset_v2_6(Dataset):
    def __init__(
        self,
        sequences: List[dict],
        sequence_dir: Path,
        cfg: Config,
        augment: bool,
        global_cache: Optional[dict] = None,
    ):
        self.sequences = sequences
        self.sequence_dir = Path(sequence_dir)
        self.cfg = cfg
        self.augment = augment
        self.global_cache = global_cache or {}

        self.video_aug = VideoAugment(cfg) if augment else None
        self.to_tensor = VideoToTensorNormalize(cfg)

    def __len__(self) -> int:
        return len(self.sequences)

    def _cache_key(self, seq_info: dict) -> str:
        seq_id = seq_info.get("sequence_id")
        animal = seq_info.get("animal", seq_info.get("animal_id", "unknown"))
        moment = seq_info.get("moment", "unknown")
        return f"{seq_id}_{animal}_{moment}"

    def _find_frames_path(self, seq_info: dict) -> Optional[Path]:
        if "sequence_path" in seq_info:
            seq_path = self.sequence_dir / seq_info["sequence_path"]
        elif "sequence_id" in seq_info:
            seq_path = self.sequence_dir / seq_info["sequence_id"]
        else:
            return None

        if seq_path.exists():
            frames = sorted(list(seq_path.glob("*.jpg")) + list(seq_path.glob("*.png")))
            if frames:
                return seq_path

        for subdir_name in ["sequence_001", "frames", "images"]:
            subdir = seq_path / subdir_name
            if subdir.exists():
                frames = sorted(list(subdir.glob("*.jpg")) + list(subdir.glob("*.png")))
                if frames:
                    return subdir

        return None

    def __getitem__(self, idx: int):
        seq_info = self.sequences[idx]
        cache_key = self._cache_key(seq_info)

        if cache_key in self.global_cache:
            frame_dir = self.global_cache[cache_key].get("path")
            frame_files = self.global_cache[cache_key].get("files")
        else:
            frame_dir = self._find_frames_path(seq_info)
            if frame_dir and frame_dir.exists():
                frame_files = sorted(list(frame_dir.glob("*.jpg")) + list(frame_dir.glob("*.png")))
                frame_files = frame_files if frame_files else None
            else:
                frame_files = None
            self.global_cache[cache_key] = {"path": frame_dir, "files": frame_files}

        moment = seq_info.get("moment", "unknown")
        animal = seq_info.get("animal", seq_info.get("animal_id", "unknown"))

        # Labels
        y1 = moment_to_task1_binary(moment)
        y2 = moment_to_task2_4class(moment)

        # Load frames
        if frame_files is None or len(frame_files) == 0:
            dummy = Image.new("RGB", self.cfg.resolution, color="black")
            frames = [dummy] * self.cfg.max_frames
        else:
            if len(frame_files) > self.cfg.max_frames:
                indices = np.linspace(0, len(frame_files) - 1, self.cfg.max_frames, dtype=int)
                selected = [frame_files[i] for i in indices]
            else:
                selected = frame_files[:]
                if len(selected) < self.cfg.max_frames:
                    selected = selected + [selected[-1]] * (self.cfg.max_frames - len(selected))

            frames = []
            last_ok = None
            for fp in selected:
                try:
                    img = Image.open(fp).convert("RGB")
                    frames.append(img)
                    last_ok = img
                except Exception:
                    frames.append(last_ok if last_ok is not None else Image.new("RGB", self.cfg.resolution, color="black"))

        if self.video_aug is not None:
            frames = self.video_aug(frames)

        x = self.to_tensor(frames)  # (T,C,H,W)

        # Random erasing on tensor (same erase per frame is complex; per-frame still helps robustness)
        if self.augment and self.cfg.use_random_erasing and random.random() < self.cfg.random_erasing_prob:
            for t in range(x.shape[0]):
                x[t] = torchvision.transforms.RandomErasing(p=1.0, scale=(0.02, 0.12), ratio=(0.3, 3.3))(x[t])

        meta = {"animal": animal, "moment": moment, "sequence_id": seq_info.get("sequence_id", f"seq_{idx}")}
        labels = {
            "pain_binary": torch.tensor(float(y1), dtype=torch.float32),
            "moment_4class": torch.tensor(int(y2), dtype=torch.long),
        }
        return x, labels, meta


# ----------------------------
# Model (pretrained ResNet18 + LSTM + attention)
# ----------------------------
class AttentionLayer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # lstm_out: (B,T,H)
        w = torch.softmax(self.attn(lstm_out), dim=1)  # (B,T,1)
        ctx = torch.sum(w * lstm_out, dim=1)  # (B,H)
        return ctx, w


class MixStyle(nn.Module):
    """
    MixStyle (Zhou et al., ICLR 2021) - mixes feature statistics to simulate domain shift.
    """

    def __init__(self, p: float = 0.5, alpha: float = 0.3, eps: float = 1e-6):
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training) or (random.random() > self.p):
            return x

        # x: (B, C, H, W)
        B = x.size(0)
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True, unbiased=False)
        sig = torch.sqrt(var + self.eps)

        x_normed = (x - mu) / sig

        # shuffle
        perm = torch.randperm(B, device=x.device)
        mu2, sig2 = mu[perm], sig[perm]

        # sample lambda from Beta(alpha, alpha)
        lam = np.random.beta(self.alpha, self.alpha)
        lam = torch.tensor(lam, device=x.device, dtype=x.dtype).view(1, 1, 1, 1)

        mu_mix = lam * mu + (1 - lam) * mu2
        sig_mix = lam * sig + (1 - lam) * sig2
        return x_normed * sig_mix + mu_mix


class TemporalPainModel_v2_6(nn.Module):
    def __init__(self, cfg: Config, use_mixstyle: bool = True):
        super().__init__()
        self.cfg = cfg

        weights = ResNet18_Weights.DEFAULT
        backbone = torchvision.models.resnet18(weights=weights)

        # Keep conv stem + layers; drop avgpool+fc, we will pool manually.
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.use_mixstyle = use_mixstyle
        self.mixstyle = MixStyle(p=0.5, alpha=0.3) if use_mixstyle else nn.Identity()

        self.cnn_out = 512
        self.lstm = nn.LSTM(
            input_size=self.cnn_out,
            hidden_size=cfg.lstm_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.attn = AttentionLayer(cfg.lstm_hidden_size)
        self.dropout = nn.Dropout(cfg.dropout)

        # Task heads
        self.head_task1 = nn.Sequential(
            nn.Linear(cfg.lstm_hidden_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(128, 1),
        )
        self.head_task2 = nn.Sequential(
            nn.Linear(cfg.lstm_hidden_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(128, 4),  # 4-class
        )

    def forward(self, x: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        # x: (B,T,C,H,W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)

        x = self.stem(x)
        x = self.layer1(x)
        x = self.mixstyle(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.pool(x).view(B * T, -1)  # (B*T, 512)
        x = x.view(B, T, -1)  # (B,T,512)

        lstm_out, _ = self.lstm(x)
        ctx, attn_w = self.attn(lstm_out)
        ctx = self.dropout(ctx)

        out = {
            "pain_binary_logits": self.head_task1(ctx).squeeze(-1),
            "moment_4class_logits": self.head_task2(ctx),
        }
        return out, attn_w


# ----------------------------
# Losses (CE + optional class-balanced focal)
# ----------------------------
def effective_num(n: int, beta: float) -> float:
    if n <= 0:
        return 0.0
    return (1.0 - beta**n) / (1.0 - beta)


def class_balanced_weights(class_counts: List[int], beta: float) -> torch.Tensor:
    eff = np.array([effective_num(n, beta) for n in class_counts], dtype=np.float64)
    eff = np.maximum(eff, 1e-8)
    w = 1.0 / eff
    w = w / w.sum() * len(class_counts)  # normalize
    return torch.tensor(w, dtype=torch.float32)


class ClassBalancedFocalLoss(nn.Module):
    def __init__(
        self,
        class_counts: List[int],
        beta: float = 0.9999,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.cb_w = class_balanced_weights(class_counts, beta=beta)  # (K,)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: (B,K), target: (B,)
        K = logits.size(1)
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        # one-hot with optional label smoothing
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.label_smoothing / (K - 1) if K > 1 else 0.0)
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.label_smoothing)

        # focal factor
        p_t = (probs * true_dist).sum(dim=1).clamp(min=1e-8)
        focal = (1.0 - p_t) ** self.gamma

        # per-sample CE
        ce = -(true_dist * log_probs).sum(dim=1)

        # class-balanced weight by true class
        w = self.cb_w.to(logits.device)[target]
        return (w * focal * ce).mean()


# ----------------------------
# Metrics helpers
# ----------------------------
@torch.no_grad()
def compute_accuracy(pred: np.ndarray, true: np.ndarray) -> float:
    return float((pred == true).mean()) if len(true) else 0.0


def moment_weight_for_sampler(moment: str) -> float:
    # oversample hard/rare moments
    return {"M0": 1.0, "M1": 1.0, "M2": 2.0, "M3": 1.5, "M4": 1.5}.get(moment, 1.0)


# ----------------------------
# Main training entry
# ----------------------------
def main():
    cfg = Config()
    seed_everything(42)

    print("=" * 80)
    print("TRAINING v2.6 (Classification improved v2.5)")
    print("Task1: Binary Pain vs No Pain")
    print("Task2: 4-class moment-intensity (M0/M1, M2, M3, M4)")
    print("Backbone: pretrained ResNet18 + LSTM + attention")
    print("=" * 80)

    # Mount Drive (Colab)
    try:
        from google.colab import drive  # type: ignore

        drive.mount("/content/drive")
        base_path = Path("/content/drive/MyDrive")
    except Exception:
        base_path = Path(os.getcwd()).parent
        print(f"Running locally - using base_path: {base_path}")

    project_dir = base_path / cfg.project_dirname
    sequence_dir = base_path / "sequence"
    checkpoint_dir = project_dir / cfg.checkpoint_subdir
    results_dir = project_dir / cfg.results_subdir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    splits_file = project_dir / "train_val_test_splits_v2.json"
    mapping_file = project_dir / "sequence_label_mapping_v2.json"
    if not splits_file.exists() or not mapping_file.exists():
        raise FileNotFoundError(
            f"Missing splits/mapping in {project_dir}. "
            f"Expected: {splits_file.name} and {mapping_file.name}."
        )

    with open(splits_file, "r") as f:
        splits = json.load(f)
    with open(mapping_file, "r") as f:
        sequence_mapping = json.load(f)

    if isinstance(sequence_mapping, dict):
        if "sequences" in sequence_mapping:
            all_sequences = sequence_mapping["sequences"]
        else:
            all_sequences = [{"sequence_id": k, **v} for k, v in sequence_mapping.items()]
    else:
        all_sequences = sequence_mapping

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Device: {device}")

    # Persist config for reproducibility
    cfg_path = results_dir / "config_v2.6.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg.__dict__, f, indent=2)
    print(f"✅ Saved config: {cfg_path}")

    # Train folds
    fold_summaries = []

    for fold_idx in range(cfg.num_folds):
        print("\n" + "=" * 80)
        print(f"Fold {fold_idx}/{cfg.num_folds - 1}")
        print("=" * 80)

        fold = splits["folds"][fold_idx]
        train_animals = fold["train_animals"]
        val_animals = fold["val_animals"]

        train_seqs = [s for s in all_sequences if s.get("animal", s.get("animal_id")) in train_animals]
        val_seqs = [s for s in all_sequences if s.get("animal", s.get("animal_id")) in val_animals]

        # Build sampler (moment-aware)
        sampler = None
        if cfg.use_weighted_sampler:
            weights = []
            for s in train_seqs:
                weights.append(moment_weight_for_sampler(s.get("moment", "unknown")))
            sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

        # Global cache for frame paths (speeds up dataset access)
        global_cache = {}

        train_ds = FacialPainDataset_v2_6(train_seqs, sequence_dir, cfg, augment=True, global_cache=global_cache)
        val_ds = FacialPainDataset_v2_6(val_seqs, sequence_dir, cfg, augment=False, global_cache=global_cache)

        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )

        # Class counts for Task2
        t2_labels = [moment_to_task2_4class(s.get("moment", "unknown")) for s in train_seqs]
        t2_counts = [int((np.array(t2_labels) == k).sum()) for k in range(4)]
        print(f"Task2 train class counts: {t2_counts} (0:M0/M1,1:M2,2:M3,3:M4)")

        # pos_weight for Task1 BCE (pain vs no pain)
        t1_labels = np.array([moment_to_task1_binary(s.get("moment", "unknown")) for s in train_seqs], dtype=np.int64)
        pos = float(t1_labels.sum())
        neg = float(len(t1_labels) - t1_labels.sum())
        pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)

        model = TemporalPainModel_v2_6(cfg, use_mixstyle=True).to(device)
        optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

        # Losses
        loss_task1 = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        if cfg.use_focal:
            loss_task2 = ClassBalancedFocalLoss(
                class_counts=t2_counts,
                beta=cfg.cb_beta if cfg.use_class_balanced else 0.0,
                gamma=cfg.focal_gamma,
                label_smoothing=cfg.label_smoothing,
            )
        else:
            w = class_balanced_weights(t2_counts, beta=cfg.cb_beta) if cfg.use_class_balanced else None
            loss_task2 = nn.CrossEntropyLoss(weight=(w.to(device) if w is not None else None), label_smoothing=cfg.label_smoothing)

        best_val = float("inf")
        best_path = checkpoint_dir / f"best_model_v2.6_fold_{fold_idx}.pt"
        history = []

        for epoch in range(cfg.num_epochs):
            model.train()
            train_loss = 0.0
            n_train = 0

            # Warmup LR (simple)
            if epoch < cfg.warmup_epochs:
                warm_lr = cfg.lr * (epoch + 1) / cfg.warmup_epochs
                for pg in optimizer.param_groups:
                    pg["lr"] = warm_lr

            for x, y, _meta in tqdm(train_loader, desc=f"Train ep {epoch+1}/{cfg.num_epochs}", leave=False):
                x = x.to(device)  # (B,T,C,H,W)
                y1 = y["pain_binary"].to(device)
                y2 = y["moment_4class"].to(device)

                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    out, _ = model(x)
                    l1 = loss_task1(out["pain_binary_logits"], y1)
                    l2 = loss_task2(out["moment_4class_logits"], y2)
                    loss = l1 + l2

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()

                train_loss += float(loss.item()) * x.size(0)
                n_train += x.size(0)

            train_loss = train_loss / max(n_train, 1)

            # Validation
            model.eval()
            val_loss = 0.0
            n_val = 0
            t1_pred, t1_true = [], []
            t2_pred, t2_true = [], []
            t2_moments = []

            with torch.no_grad():
                for x, y, meta in tqdm(val_loader, desc="Val", leave=False):
                    x = x.to(device)
                    y1 = y["pain_binary"].to(device)
                    y2 = y["moment_4class"].to(device)

                    with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                        out, _ = model(x)
                        l1 = loss_task1(out["pain_binary_logits"], y1)
                        l2 = loss_task2(out["moment_4class_logits"], y2)
                        loss = l1 + l2

                    val_loss += float(loss.item()) * x.size(0)
                    n_val += x.size(0)

                    # preds
                    p1 = (torch.sigmoid(out["pain_binary_logits"]) > 0.5).long().cpu().numpy()
                    p2 = torch.argmax(out["moment_4class_logits"], dim=1).cpu().numpy()
                    t1_pred.extend(p1.tolist())
                    t1_true.extend(y1.long().cpu().numpy().tolist())
                    t2_pred.extend(p2.tolist())
                    t2_true.extend(y2.cpu().numpy().tolist())
                    t2_moments.extend(meta["moment"])

            val_loss = val_loss / max(n_val, 1)
            scheduler.step()

            t1_acc = compute_accuracy(np.array(t1_pred), np.array(t1_true))
            t2_acc = compute_accuracy(np.array(t2_pred), np.array(t2_true))

            row = {
                "fold": fold_idx,
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "task1_acc": t1_acc,
                "task2_acc": t2_acc,
                "lr": optimizer.param_groups[0]["lr"],
            }
            history.append(row)
            print(
                f"Epoch {epoch+1:03d} | train {train_loss:.4f} | val {val_loss:.4f} | "
                f"T1 acc {t1_acc:.3f} | T2 acc {t2_acc:.3f}"
            )

            if val_loss < best_val:
                best_val = val_loss
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "cfg": cfg.__dict__,
                        "fold": fold_idx,
                        "epoch": epoch + 1,
                        "val_loss": best_val,
                        "t2_counts": t2_counts,
                    },
                    best_path,
                )

        # Save fold history
        hist_df = pd.DataFrame(history)
        hist_csv = results_dir / f"training_history_v2.6_fold_{fold_idx}.csv"
        hist_df.to_csv(hist_csv, index=False)
        print(f"✅ Saved fold history: {hist_csv}")
        print(f"✅ Best model: {best_path.name} (val_loss={best_val:.4f})")

        fold_summaries.append({"fold": fold_idx, "best_val_loss": best_val, "best_model": best_path.name})

    # Save summary
    summary_df = pd.DataFrame(fold_summaries)
    summary_path = results_dir / "fold_summary_v2.6.csv"
    summary_df.to_csv(summary_path, index=False)
    print("\n✅ Training complete.")
    print(f"✅ Saved summary: {summary_path}")


if __name__ == "__main__":
    main()

