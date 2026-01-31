# ============================================================================
# TEST SET EVALUATION - v2.6 (Binary + 4-Class)
# Evaluates models on held-out test set (default animals 14 and 17)
# ============================================================================

import os
import json
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as TF
from torchvision.models import ResNet18_Weights

from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
from scipy.stats import mode

warnings.filterwarnings("ignore")


def moment_to_task1_binary(moment: str) -> int:
    return 1 if moment in ["M2", "M3", "M4"] else 0


def moment_to_task2_4class(moment: str) -> int:
    if moment in ["M0", "M1"]:
        return 0
    if moment == "M2":
        return 1
    if moment == "M3":
        return 2
    if moment == "M4":
        return 3
    return 0


class AttentionLayer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out: torch.Tensor):
        w = torch.softmax(self.attn(lstm_out), dim=1)
        ctx = torch.sum(w * lstm_out, dim=1)
        return ctx, w


class MixStyle(nn.Module):
    def __init__(self, p: float = 0.0, alpha: float = 0.3, eps: float = 1e-6):
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Disabled at eval by p=0.0
        return x


class TemporalPainModel_v2_6(nn.Module):
    def __init__(self, lstm_hidden_size: int = 256, dropout: float = 0.3):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT
        backbone = torchvision.models.resnet18(weights=weights)

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.mixstyle = MixStyle(p=0.0)

        self.cnn_out = 512
        self.lstm = nn.LSTM(
            input_size=self.cnn_out,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.attn = AttentionLayer(lstm_hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.head_task1 = nn.Sequential(
            nn.Linear(lstm_hidden_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
        self.head_task2 = nn.Sequential(
            nn.Linear(lstm_hidden_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 4),
        )

    def forward(self, x: torch.Tensor):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)

        x = self.stem(x)
        x = self.layer1(x)
        x = self.mixstyle(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.pool(x).view(B * T, -1)
        x = x.view(B, T, -1)

        lstm_out, _ = self.lstm(x)
        ctx, attn_w = self.attn(lstm_out)
        ctx = self.dropout(ctx)

        out = {
            "pain_binary_logits": self.head_task1(ctx).squeeze(-1),
            "moment_4class_logits": self.head_task2(ctx),
        }
        return out, attn_w


class VideoToTensorNormalize:
    def __init__(self, resolution=(112, 112)):
        self.resolution = resolution
        weights = ResNet18_Weights.DEFAULT
        self.mean = weights.transforms().mean
        self.std = weights.transforms().std

    def __call__(self, frames):
        ts = []
        for img in frames:
            img = TF.resize(img, self.resolution)
            t = TF.to_tensor(img)
            t = TF.normalize(t, mean=self.mean, std=self.std)
            ts.append(t)
        return torch.stack(ts, dim=0)


class FacialPainDataset_v2_6(torch.utils.data.Dataset):
    def __init__(self, sequences, sequence_dir, max_frames=32, resolution=(112, 112), global_cache=None):
        self.sequences = sequences
        self.sequence_dir = Path(sequence_dir)
        self.max_frames = max_frames
        self.global_cache = global_cache or {}
        self.to_tensor = VideoToTensorNormalize(resolution=resolution)

    def __len__(self):
        return len(self.sequences)

    def _cache_key(self, seq_info):
        seq_id = seq_info.get("sequence_id")
        animal = seq_info.get("animal", seq_info.get("animal_id", "unknown"))
        moment = seq_info.get("moment", "unknown")
        return f"{seq_id}_{animal}_{moment}"

    def _find_frames_path(self, seq_info):
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

    def __getitem__(self, idx):
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

        y1 = moment_to_task1_binary(moment)
        y2 = moment_to_task2_4class(moment)

        if frame_files is None or len(frame_files) == 0:
            dummy = Image.new("RGB", (112, 112), color="black")
            frames = [dummy] * self.max_frames
        else:
            if len(frame_files) > self.max_frames:
                indices = np.linspace(0, len(frame_files) - 1, self.max_frames, dtype=int)
                selected = [frame_files[i] for i in indices]
            else:
                selected = frame_files[:]
                if len(selected) < self.max_frames:
                    selected = selected + [selected[-1]] * (self.max_frames - len(selected))

            frames = []
            last_ok = None
            for fp in selected:
                try:
                    img = Image.open(fp).convert("RGB")
                    frames.append(img)
                    last_ok = img
                except Exception:
                    frames.append(last_ok if last_ok is not None else Image.new("RGB", (112, 112), color="black"))

        x = self.to_tensor(frames)
        labels = {
            "pain_binary": torch.tensor(float(y1), dtype=torch.float32),
            "moment_4class": torch.tensor(int(y2), dtype=torch.long),
        }
        meta = {"animal": animal, "moment": moment, "sequence_id": seq_info.get("sequence_id", f"seq_{idx}")}
        return x, labels, meta


@torch.no_grad()
def evaluate_fold(model, loader, device):
    model.eval()
    pain_logits = []
    moment_logits = []
    y1_all = []
    y2_all = []
    moments = []
    animals = []

    for x, y, meta in loader:
        x = x.to(device)
        out, _ = model(x)
        pain_logits.append(out["pain_binary_logits"].detach().cpu())
        moment_logits.append(out["moment_4class_logits"].detach().cpu())
        y1_all.append(y["pain_binary"])
        y2_all.append(y["moment_4class"])
        moments.extend(meta["moment"])
        animals.extend(meta["animal"])

    pain_logits = torch.cat(pain_logits).numpy()
    moment_logits = torch.cat(moment_logits).numpy()
    y1_all = torch.cat(y1_all).numpy().astype(int)
    y2_all = torch.cat(y2_all).numpy().astype(int)

    pain_pred = (1 / (1 + np.exp(-pain_logits)) > 0.5).astype(int)
    moment_pred = np.argmax(moment_logits, axis=1)

    return pain_pred, y1_all, moment_pred, y2_all, moments, animals, pain_logits, moment_logits


def main():
    print("=" * 80)
    print("TEST SET EVALUATION - v2.6 (Binary + 4-Class)")
    print("=" * 80)

    # Mount Drive (Colab)
    try:
        from google.colab import drive  # type: ignore

        drive.mount("/content/drive")
        base_path = Path("/content/drive/MyDrive")
    except Exception:
        base_path = Path(os.getcwd()).parent
        print(f"Running locally - using base_path: {base_path}")

    project_dir = base_path / "facial_pain_project_v2"
    sequence_dir = base_path / "sequence"
    checkpoint_dir = project_dir / "checkpoints_v2.6"
    results_dir = project_dir / "results_v2.6"
    results_dir.mkdir(parents=True, exist_ok=True)

    splits_file = project_dir / "train_val_test_splits_v2.json"
    mapping_file = project_dir / "sequence_label_mapping_v2.json"
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

    test_animals = splits.get("test_animals", [14, 17])
    test_seqs = [s for s in all_sequences if s.get("animal", s.get("animal_id")) in test_animals]
    print(f"Test animals: {test_animals} | sequences: {len(test_seqs)}")

    # global cache for paths
    global_cache = {}

    ds = FacialPainDataset_v2_6(test_seqs, sequence_dir, max_frames=32, resolution=(112, 112), global_cache=global_cache)
    loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Device: {device}")

    # Evaluate per fold models
    fold_results = []
    available = []
    for fold_idx in range(9):
        p = checkpoint_dir / f"best_model_v2.6_fold_{fold_idx}.pt"
        if p.exists():
            available.append(fold_idx)
    print(f"Found folds: {available}")

    for fold_idx in available:
        ckpt_path = checkpoint_dir / f"best_model_v2.6_fold_{fold_idx}.pt"
        ckpt = torch.load(ckpt_path, map_location=device)
        cfg = ckpt.get("cfg", {})
        model = TemporalPainModel_v2_6(
            lstm_hidden_size=int(cfg.get("lstm_hidden_size", 256)),
            dropout=float(cfg.get("dropout", 0.3)),
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])

        pain_pred, pain_true, m_pred, m_true, moments, animals, pain_logits, m_logits = evaluate_fold(model, loader, device)

        t1 = {
            "fold": fold_idx,
            "task": "task1_binary",
            "accuracy": accuracy_score(pain_true, pain_pred),
            "f1": f1_score(pain_true, pain_pred, zero_division=0),
            "precision": precision_score(pain_true, pain_pred, zero_division=0),
            "recall": recall_score(pain_true, pain_pred, zero_division=0),
            "n": int(len(pain_true)),
        }
        t2 = {
            "fold": fold_idx,
            "task": "task2_4class",
            "accuracy": accuracy_score(m_true, m_pred),
            "f1_weighted": f1_score(m_true, m_pred, average="weighted", zero_division=0),
            "f1_macro": f1_score(m_true, m_pred, average="macro", zero_division=0),
            "n": int(len(m_true)),
        }
        fold_results.append((t1, t2, pain_pred, m_pred, pain_logits, m_logits, pain_true, m_true, moments, animals))
        print(f"Fold {fold_idx}: T1 acc={t1['accuracy']:.3f} F1={t1['f1']:.3f} | T2 acc={t2['accuracy']:.3f} F1w={t2['f1_weighted']:.3f}")

    if not fold_results:
        raise RuntimeError(f"No v2.6 checkpoints found in {checkpoint_dir}")

    # Ensemble (majority vote; also average logits for stability)
    pain_preds_stack = np.stack([fr[2] for fr in fold_results], axis=0)
    m_preds_stack = np.stack([fr[3] for fr in fold_results], axis=0)

    ensemble_pain = np.round(pain_preds_stack.mean(axis=0)).astype(int)
    ensemble_m = mode(m_preds_stack, axis=0)[0].flatten()

    pain_true = fold_results[0][6]
    m_true = fold_results[0][7]
    moments = fold_results[0][8]
    animals = fold_results[0][9]

    # Metrics
    class_names = ["NoPain (M0/M1)", "Acute (M2)", "Declining (M3)", "Recovery (M4)"]

    ensemble_task1 = {
        "accuracy": accuracy_score(pain_true, ensemble_pain),
        "f1": f1_score(pain_true, ensemble_pain, zero_division=0),
        "precision": precision_score(pain_true, ensemble_pain, zero_division=0),
        "recall": recall_score(pain_true, ensemble_pain, zero_division=0),
        "n": int(len(pain_true)),
    }
    ensemble_task2 = {
        "accuracy": accuracy_score(m_true, ensemble_m),
        "f1_weighted": f1_score(m_true, ensemble_m, average="weighted", zero_division=0),
        "f1_macro": f1_score(m_true, ensemble_m, average="macro", zero_division=0),
        "n": int(len(m_true)),
    }

    print("\n" + "=" * 80)
    print("ENSEMBLE RESULTS")
    print("=" * 80)
    print("Task1 (binary):", ensemble_task1)
    print("Task2 (4-class):", ensemble_task2)
    print("\nTask2 classification report:")
    print(classification_report(m_true, ensemble_m, target_names=class_names, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(m_true, ensemble_m, labels=[0, 1, 2, 3])
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    # Moment-wise breakdown (still report by raw moments)
    moment_rows = []
    for mom in ["M0", "M1", "M2", "M3", "M4"]:
        mask = np.array(moments) == mom
        if mask.sum() == 0:
            continue
        moment_rows.append(
            {
                "moment": mom,
                "n": int(mask.sum()),
                "task1_acc": accuracy_score(pain_true[mask], ensemble_pain[mask]),
                "task2_acc": accuracy_score(m_true[mask], ensemble_m[mask]),
            }
        )
    moment_df = pd.DataFrame(moment_rows)

    # Save CSVs
    per_fold_rows = []
    for (t1, t2, *_rest) in fold_results:
        per_fold_rows.append({**t1})
        per_fold_rows.append({**t2})
    per_fold_df = pd.DataFrame(per_fold_rows)

    out_fold = results_dir / "test_eval_v2.6_per_fold.csv"
    out_ens = results_dir / "test_eval_v2.6_ensemble.json"
    out_cm = results_dir / "test_eval_v2.6_confusion_matrix.csv"
    out_mom = results_dir / "test_eval_v2.6_moment_wise.csv"

    per_fold_df.to_csv(out_fold, index=False)
    cm_df.to_csv(out_cm, index=True)
    moment_df.to_csv(out_mom, index=False)
    with open(out_ens, "w") as f:
        json.dump({"task1": ensemble_task1, "task2": ensemble_task2}, f, indent=2)

    print(f"\n✅ Saved: {out_fold}")
    print(f"✅ Saved: {out_mom}")
    print(f"✅ Saved: {out_cm}")
    print(f"✅ Saved: {out_ens}")


if __name__ == "__main__":
    main()

