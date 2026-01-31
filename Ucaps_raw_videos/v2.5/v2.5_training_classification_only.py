# ============================================================================
# PHASE 2: MODEL TRAINING (v2.5 - DUAL CLASSIFICATION FOR T4 GPU)
#
# Key Features:
# - DUAL CLASSIFICATION TASKS:
#   1) Binary: Pain vs No Pain (M0/M1=0, M2/M3/M4=1)
#   2) 3-Class: Intensity Moment (M0/M1=0, M2=1, M3/M4=2)
# - 2D CNN + LSTM: More suitable for small datasets than 3D CNN
# - T4 GPU OPTIMIZED: Smaller batch size, fewer workers
# - MOMENT-WEIGHTED LOSS: Based on v2.4 insights (M2 challenging, M3 best)
# - METRICS: F1-Score, Accuracy, Precision, Recall for both tasks
#
# v2.5 Improvements (Based on v2.4 Results):
# - Dual classification tasks (binary + 3-class)
# - 2D CNN + LSTM architecture (validated in v2.4)
# - Moment weights adjusted based on v2.4 test performance
# - Optimized for T4 GPU (16GB VRAM)
# - Weighted loss for both tasks with moment-based weighting
# - Comprehensive classification metrics for both tasks
#
# v2.4 Insights Incorporated:
# - M2 (peak pain) is challenging (negative correlation in test)
# - M3 performs best (R²=0.4209, r=0.9620)
# - Moment weights: M2=4.0 (reduced from 10.0), M3=2.0
# - Best features: Orbital_tightening, Ears_frontal, Ears_lateral
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

import json
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from collections import defaultdict, Counter
import warnings
import os
from typing import Dict, List, Optional
warnings.filterwarnings('ignore')

# ============================================================================
# RESUME SETTINGS (Colab disconnect-safe)
# ============================================================================
# Best practice for Colab: avoid interactive input() so runs can be restarted
# and continue automatically after disconnect.
#
# You can override START_FOLD via environment variable START_FOLD="3"
# ============================================================================
AUTO_RESUME = True
START_FOLD = int(os.environ.get("START_FOLD", "0"))

# ============================================================================
# SETUP: Mount Drive and Verify Paths
# ============================================================================
print("="*80)
print("PHASE 2: MODEL TRAINING (v2.5 - DUAL CLASSIFICATION FOR T4 GPU)")
print("="*80)
print("Task 1: Binary Classification (Pain vs No Pain)")
print("Task 2: 3-Class Classification (Intensity Moment)")
print("Architecture: 2D CNN + LSTM (validated in v2.4)")
print("GPU: T4 Optimized (16GB VRAM)")
print("="*80)

# Mount Drive (for Colab)
try:
    from google.colab import drive
    drive.mount('/content/drive')
    base_path = Path('/content/drive/MyDrive')
except ImportError:
    # Running locally
    base_path = Path(os.getcwd()).parent
    print(f"Running locally - using base_path: {base_path}")

project_dir = base_path / 'facial_pain_project_v2'
sequence_dir = base_path / 'sequence'
checkpoint_dir = project_dir / 'checkpoints_v2.5'
checkpoint_dir.mkdir(exist_ok=True, parents=True)

# Verify paths exist
print(f"\n📁 Checking paths...")
print(f"   Project dir: {project_dir} {'✅' if project_dir.exists() else '❌'}")
print(f"   Sequence dir: {sequence_dir} {'✅' if sequence_dir.exists() else '❌'}")

# Find sequence directory if not at expected location
if not sequence_dir.exists():
    print("\n⚠️  sequence/ not found at expected location. Searching...")
    possible_paths = [
        base_path / 'sequence',
        base_path / 'sequences',
        base_path / 'VIDEOS FACIAL BOVINE' / 'sequence',
        base_path / 'facial_pain_project_v2' / 'sequence',
    ]
    for path in possible_paths:
        if path.exists():
            sequence_dir = path
            print(f"   ✅ Found at: {sequence_dir}")
            break
    else:
        raise FileNotFoundError(f"Could not find sequence directory. Checked: {possible_paths}")

# Load splits and mappings
splits_file = project_dir / 'train_val_test_splits_v2.json'
mapping_file = project_dir / 'sequence_label_mapping_v2.json'

if not splits_file.exists():
    raise FileNotFoundError(f"Missing: {splits_file}")
if not mapping_file.exists():
    raise FileNotFoundError(f"Missing: {mapping_file}")

print(f"   Splits file: {'✅' if splits_file.exists() else '❌'}")
print(f"   Mapping file: {'✅' if mapping_file.exists() else '❌'}")

with open(splits_file, 'r') as f:
    splits = json.load(f)
with open(mapping_file, 'r') as f:
    sequence_mapping = json.load(f)

# Handle both list and dict formats
if isinstance(sequence_mapping, dict):
    if 'sequences' in sequence_mapping:
        all_sequences = sequence_mapping['sequences']
    else:
        all_sequences = [{'sequence_id': k, **v} for k, v in sequence_mapping.items()]
else:
    all_sequences = sequence_mapping

print(f"\n✅ Loaded {len(all_sequences)} sequences")
print(f"✅ Checkpoint dir: {checkpoint_dir}")

# ============================================================================
# 1. MODEL ARCHITECTURE (v2.5 - 2D CNN + LSTM, Classification Only)
# ============================================================================
print("\n[1] Defining model architecture (2D CNN + LSTM, Classification Only)...")

class AttentionLayer(nn.Module):
    """Attention mechanism for temporal aggregation"""
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output):
        # lstm_output: (batch, seq_len, hidden_size)
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        return context_vector, attention_weights


class TemporalPainModel_v2_5(nn.Module):
    """
    v2.5: 2D CNN + LSTM architecture for dual classification
    
    Architecture:
    - 2D CNN: Extracts spatial features from each frame individually
    - LSTM: Processes sequence of frame features (standard, not bidirectional)
    - Attention: Aggregates temporal information
    - Output: 
      1) Binary classification (Pain vs No Pain)
      2) 3-Class classification (Intensity Moment: No Pain, Acute Pain, Residual Pain)
    
    Rationale:
    - More parameter-efficient than 3D CNN (validated in v2.4)
    - Better suited for small datasets
    - Standard LSTM (not bidirectional) reduces overfitting risk
    - Optimized for T4 GPU memory constraints
    - Dual tasks provide complementary information
    """
    def __init__(self, num_frames=32, lstm_hidden_size=128, use_bidirectional=False):
        super(TemporalPainModel_v2_5, self).__init__()
        
        # 2D CNN for spatial feature extraction
        # Input: (batch, frames, 3, H, W) -> process each frame
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Global Average Pooling
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # CNN output size (after global avg pool)
        self.cnn_output_size = 256
        
        # LSTM for temporal modeling
        self.use_bidirectional = use_bidirectional
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_size,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=use_bidirectional
        )
        
        self.lstm_output_size = lstm_hidden_size * 2 if use_bidirectional else lstm_hidden_size
        
        # Attention
        self.attention = AttentionLayer(self.lstm_output_size)
        
        # Task 1: Binary classification head (Pain vs No Pain)
        self.pain_classification_head = nn.Sequential(
            nn.Linear(self.lstm_output_size, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1)  # Binary classification (logits)
        )
        
        # Task 2: 3-Class classification head (Intensity Moment)
        self.intensity_classification_head = nn.Sequential(
            nn.Linear(self.lstm_output_size, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 3)  # 3-class classification (logits): No Pain, Acute Pain, Residual Pain
        )
        
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x: (batch, frames, 3, H, W)
        batch_size, num_frames, C, H, W = x.size()
        
        # Process each frame through 2D CNN
        # Reshape to (batch * frames, 3, H, W)
        x = x.view(batch_size * num_frames, C, H, W)
        
        # 2D CNN feature extraction
        cnn_features = self.cnn(x)  # (batch * frames, 256, 1, 1)
        cnn_features = cnn_features.view(batch_size * num_frames, -1)  # (batch * frames, 256)
        
        # Reshape back to (batch, frames, 256) for LSTM
        cnn_features = cnn_features.view(batch_size, num_frames, self.cnn_output_size)
        
        # LSTM for temporal modeling
        lstm_out, _ = self.lstm(cnn_features)  # (batch, frames, hidden_size)
        
        # Attention aggregation
        context, attention_weights = self.attention(lstm_out)  # (batch, hidden_size)
        context = self.dropout(context)
        
        # Dual classification outputs
        outputs = {}
        # Task 1: Binary classification (Pain vs No Pain)
        outputs['pain_classification'] = self.pain_classification_head(context).squeeze(-1)
        # Task 2: 3-Class classification (Intensity Moment)
        outputs['intensity_classification'] = self.intensity_classification_head(context)
        
        return outputs, attention_weights

print("✅ Model defined (v2.5 - 2D CNN + LSTM, Dual Classification)")

# ============================================================================
# 2. DATASET CLASS (v2.5 - Classification Only)
# ============================================================================
print("\n[2] Defining dataset class...")

class FacialPainDataset_v2_5(Dataset):
    """v2.5: Dual classification - binary Pain vs No Pain + 3-class Intensity Moment"""
    def __init__(self, sequence_mapping, sequence_dir, max_frames=32,
                 transform=None, augment=False):
        self.sequence_mapping = sequence_mapping
        self.sequence_dir = Path(sequence_dir)
        self.max_frames = max_frames
        self.transform = transform
        self.augment = augment

        print("   Pre-computing frame paths and file lists (one-time)...")
        self.frame_paths_cache = {}
        self.frame_files_cache = {}
        for idx, seq_info in enumerate(tqdm(sequence_mapping, desc="Caching paths")):
            frame_path = self._find_frames_path(seq_info)
            self.frame_paths_cache[idx] = frame_path
            if frame_path and frame_path.exists():
                frame_files = sorted(list(frame_path.glob('*.jpg')) + list(frame_path.glob('*.png')))
                self.frame_files_cache[idx] = frame_files if len(frame_files) > 0 else None
            else:
                self.frame_files_cache[idx] = None
        print(f"   ✅ Cached {len(self.frame_paths_cache)} sequence paths")

        if augment:
            print("   Applying STRONG augmentations (Affine, Blur, Jitter)")
            self.aug_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 2.0))
            ])

    def __len__(self):
        return len(self.sequence_mapping)

    def _find_frames_path(self, seq_info):
        if 'sequence_path' in seq_info:
            seq_path = self.sequence_dir / seq_info['sequence_path']
        elif 'sequence_id' in seq_info:
            seq_path = self.sequence_dir / seq_info['sequence_id']
        else:
            return None
        if seq_path.exists():
            frame_files = sorted(list(seq_path.glob('*.jpg')) + list(seq_path.glob('*.png')))
            if len(frame_files) > 0:
                return seq_path
        possible_subdirs = [seq_path / d for d in ['sequence_001', 'frames', 'images']]
        for subdir in possible_subdirs:
            if subdir.exists():
                frames = sorted(list(subdir.glob('*.jpg')) + list(subdir.glob('*.png')))
                if len(frames) > 0:
                    return subdir
        for subdir in seq_path.rglob('*'):
            if subdir.is_dir():
                frames = sorted(list(subdir.glob('*.jpg')) + list(subdir.glob('*.png')))
                if len(frames) > 0:
                    return subdir
        return None

    def __getitem__(self, idx):
        seq_info = self.sequence_mapping[idx]
        frame_dir = self.frame_paths_cache.get(idx)

        if frame_dir is None or not frame_dir.exists():
            dummy_frame = Image.new('RGB', (112, 112), color='black')
            frames = [dummy_frame] * self.max_frames
            frames_tensor = torch.stack([self.transform(img) for img in frames])

            # Dual classification labels
            labels = {}
            labels['pain_classification'] = torch.tensor(0.0, dtype=torch.float32)  # No Pain
            labels['intensity_classification'] = torch.tensor(0, dtype=torch.long)  # No Pain class

            metadata = {
                'animal': seq_info.get('animal', 'unknown'),
                'moment': seq_info.get('moment', 'unknown'),
                'sequence_id': seq_info.get('sequence_id', f'seq_{idx}')
            }
            return frames_tensor, labels, metadata

        frame_files = self.frame_files_cache.get(idx)
        if frame_files is None:
            frame_files = []

        if len(frame_files) > self.max_frames:
            indices = np.linspace(0, len(frame_files)-1, self.max_frames, dtype=int)
            frame_files = [frame_files[i] for i in indices]
        elif len(frame_files) < self.max_frames and len(frame_files) > 0:
            frame_files = frame_files + [frame_files[-1]] * (self.max_frames - len(frame_files))

        frames = []
        last_valid_img_tensor = None
        for frame_file in frame_files:
            try:
                img = Image.open(frame_file).convert('RGB')
                if self.augment and self.aug_transform:
                    img = self.aug_transform(img)
                if self.transform:
                    img_tensor = self.transform(img)
                else:
                    img_tensor = transforms.ToTensor()(img)
                frames.append(img_tensor)
                last_valid_img_tensor = img_tensor
            except Exception:
                if last_valid_img_tensor is not None:
                    frames.append(last_valid_img_tensor.clone())
                else:
                    dummy_tensor = torch.zeros(3, 112, 112, dtype=torch.float32)
                    if self.transform:
                        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                        dummy_tensor = (dummy_tensor - mean) / std
                    frames.append(dummy_tensor)
                    last_valid_img_tensor = dummy_tensor

        frames = torch.stack(frames)

        metadata = {
            'animal': seq_info.get('animal', seq_info.get('animal_id', 'unknown')),
            'moment': seq_info.get('moment', 'unknown'),
            'sequence_id': seq_info.get('sequence_id', f'seq_{idx}')
        }

        # Task 1: Binary classification label
        # M0/M1 = 0 (No Pain), M2/M3/M4 = 1 (Pain)
        moment = metadata['moment']
        is_pain = 1.0 if moment in ['M2', 'M3', 'M4'] else 0.0
        
        # Task 2: 3-Class classification label (Intensity Moment)
        # M0/M1 = 0 (No Pain), M2 = 1 (Acute Pain), M3/M4 = 2 (Residual Pain)
        if moment in ['M0', 'M1']:
            intensity_class = 0  # No Pain
        elif moment == 'M2':
            intensity_class = 1  # Acute Pain
        elif moment in ['M3', 'M4']:
            intensity_class = 2  # Residual Pain
        else:
            intensity_class = 0  # Default to No Pain
        
        labels = {}
        labels['pain_classification'] = torch.tensor(is_pain, dtype=torch.float32)
        labels['intensity_classification'] = torch.tensor(intensity_class, dtype=torch.long)

        return frames, labels, metadata

print("✅ Dataset class defined (v2.5 - Dual Classification)")

# ============================================================================
# 3. WEIGHTED CLASSIFICATION LOSS (v2.5 - Moment-Weighted BCE)
# ============================================================================
print("\n[3] Defining weighted classification loss function...")

class WeightedDualClassificationLoss(nn.Module):
    """
    v2.5: Dual classification loss with moment-based weighting
    
    Based on v2.4 insights:
    - M2 (peak pain) is challenging but critical (weight: 4.0)
    - M3 performs best in regression (weight: 2.0)
    - M0/M1 are baseline (weight: 1.0)
    - M4 is recovery assessment (weight: 1.2)
    
    Tasks:
    1) Binary: Pain vs No Pain (BCE with logits)
    2) 3-Class: Intensity Moment (CrossEntropy)
    """
    def __init__(self, moment_weights=None, task1_weight=1.0, task2_weight=1.0, 
                 pos_weight=None, intensity_class_weights=None):
        super(WeightedDualClassificationLoss, self).__init__()

        # Moment weights based on v2.4 test performance analysis
        # v2.4 Test Set Findings:
        # - M2 (Peak pain): r=-0.5715, R²=-0.6405 (NEGATIVE correlation - very challenging!)
        #   BUT it's critical for "Acute Pain" class in 3-class task
        # - M3 (Declining): R²=0.4209, r=0.9620 (BEST performer)
        # - M1 performs BETTER than M0 (MAE: 1.300 vs 1.909)
        # - M4: Recovery assessment (MAE=1.608)
        self.moment_weights = moment_weights or {
            'M0': 1.0,   # Baseline - no pain (MAE=1.909 in v2.4)
            'M1': 1.0,   # Pre-procedure - no pain (MAE=1.300, performs BETTER than M0)
            'M2': 4.0,   # CRITICAL but challenging - negative correlation in test (r=-0.5715)
                         # Reduced from 10.0 to prevent overfitting (v2.4 insight)
                         # Critical for "Acute Pain" classification
            'M3': 2.0,   # Declining pain - BEST in v2.4 (R²=0.4209, r=0.9620)
            'M4': 1.2    # Recovery assessment (MAE=1.608)
        }
        
        # Task weights (balance between binary and 3-class)
        self.task1_weight = task1_weight  # Binary classification weight
        self.task2_weight = task2_weight  # 3-Class classification weight
        
        # Positive class weight for binary task (for class imbalance)
        self.pos_weight = pos_weight
        
        # Class weights for 3-class task (for class imbalance)
        # [No Pain, Acute Pain, Residual Pain]
        self.intensity_class_weights = intensity_class_weights

        print("\n" + "="*70)
        print("DUAL CLASSIFICATION LOSS CONFIGURATION (v2.5)")
        print("="*70)
        print("\n⏰ Moment Weights (Based on v2.4 Test Performance):")
        for moment, weight in sorted(self.moment_weights.items(), key=lambda x: x[1], reverse=True):
            print(f"   {moment:25s}: {weight:.1f}×")
        print(f"\n📊 Task Weights:")
        print(f"   Task 1 (Binary Pain/No Pain): {self.task1_weight:.1f}×")
        print(f"   Task 2 (3-Class Intensity): {self.task2_weight:.1f}×")
        if self.pos_weight is not None:
            print(f"\n⚖️  Binary Task - Positive Class Weight: {self.pos_weight:.2f}")
        if self.intensity_class_weights is not None:
            print(f"\n⚖️  3-Class Task - Class Weights: {self.intensity_class_weights}")
        print("="*70 + "\n")

    def forward(self, predictions, targets, moments):
        total_loss = 0.0
        
        # Task 1: Binary Classification (Pain vs No Pain)
        if 'pain_classification' in predictions and 'pain_classification' in targets:
            pred_logits = predictions['pain_classification']
            target = targets['pain_classification']
            
            # Get moment weights
            moment_weight_tensor = torch.tensor(
                [self.moment_weights.get(m, 1.0) for m in moments],
                device=pred_logits.device, dtype=torch.float32
            )
            
            # Calculate BCE loss per-sample
            if self.pos_weight is not None:
                pos_weight_tensor = torch.tensor(
                    self.pos_weight, device=pred_logits.device, dtype=torch.float32
                )
                bce_loss = F.binary_cross_entropy_with_logits(
                    pred_logits, target, 
                    pos_weight=pos_weight_tensor,
                    reduction='none'
                )
            else:
                bce_loss = F.binary_cross_entropy_with_logits(
                    pred_logits, target, 
                    reduction='none'
                )
            
            # Apply moment weights
            task1_loss = (bce_loss * moment_weight_tensor).mean()
            total_loss += self.task1_weight * task1_loss
        
        # Task 2: 3-Class Classification (Intensity Moment)
        if 'intensity_classification' in predictions and 'intensity_classification' in targets:
            pred_logits = predictions['intensity_classification']
            target = targets['intensity_classification']  # Long tensor with class indices
            
            # Get moment weights
            moment_weight_tensor = torch.tensor(
                [self.moment_weights.get(m, 1.0) for m in moments],
                device=pred_logits.device, dtype=torch.float32
            )
            
            # Calculate CrossEntropy loss per-sample
            if self.intensity_class_weights is not None:
                class_weights = torch.tensor(
                    self.intensity_class_weights, 
                    device=pred_logits.device, 
                    dtype=torch.float32
                )
                ce_loss = F.cross_entropy(
                    pred_logits, target,
                    weight=class_weights,
                    reduction='none'
                )
            else:
                ce_loss = F.cross_entropy(
                    pred_logits, target,
                    reduction='none'
                )
            
            # Apply moment weights
            task2_loss = (ce_loss * moment_weight_tensor).mean()
            total_loss += self.task2_weight * task2_loss
        
        return total_loss

print("✅ Weighted dual classification loss function defined (v2.5)")

# ============================================================================
# 4. STRATIFIED SAMPLER
# ============================================================================
print("\n[4] Defining stratified sampler...")

def create_stratified_sampler(dataset):
    moment_counts = Counter()
    for seq_info in dataset.sequence_mapping:
        moment_counts[seq_info.get('moment', 'unknown')] += 1
    total = sum(moment_counts.values())
    moment_weights = {m: total / count for m, count in moment_counts.items()}
    sample_weights = [moment_weights.get(s.get('moment', 'unknown'), 1.0) for s in dataset.sequence_mapping]
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(dataset), replacement=True)

print("✅ Stratified sampler defined")

# ============================================================================
# 5. TRAINING FUNCTIONS (v2.5 - Classification Metrics)
# ============================================================================
print("\n[5] Defining training functions...")

def train_one_epoch(model, dataloader, criterion, optimizer, device, config=None, scaler=None):
    model.train()
    total_loss = 0.0
    num_batches = 0
    gradient_clip = config.get('gradient_clip', 0.5)
    use_amp = scaler is not None

    pbar = tqdm(dataloader, desc="Training", mininterval=1.0)
    for batch_idx, (frames, labels, metadata) in enumerate(pbar):
        frames = frames.to(device, non_blocking=True)
        moments = metadata['moment']
        targets = {k: v.to(device, non_blocking=True) for k, v in labels.items()}

        optimizer.zero_grad()

        if use_amp:
            with torch.cuda.amp.autocast():
                outputs, _ = model(frames)
                loss = criterion(outputs, targets, moments)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs, _ = model(frames)
            loss = criterion(outputs, targets, moments)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if batch_idx % max(1, len(dataloader) // 10) == 0 or batch_idx == len(dataloader) - 1:
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'avg': f'{total_loss/num_batches:.4f}'})

    return total_loss / num_batches


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    # Store classification results for metrics (Task 1: Binary)
    all_pain_preds = []
    all_pain_targets = []
    
    # Store classification results for metrics (Task 2: 3-Class)
    all_intensity_preds = []
    all_intensity_targets = []

    with torch.no_grad():
        for frames, labels, metadata in dataloader:
            frames = frames.to(device, non_blocking=True)
            moments = metadata['moment']
            targets = {k: v.to(device, non_blocking=True) for k, v in labels.items()}

            outputs, _ = model(frames)
            loss = criterion(outputs, targets, moments)

            total_loss += loss.item()
            num_batches += 1

            # Task 1: Binary classification metrics
            if 'pain_classification' in outputs and 'pain_classification' in targets:
                pred_logits = outputs['pain_classification']
                target = targets['pain_classification']
                pred_binary = (torch.sigmoid(pred_logits) > 0.5).int()
                all_pain_preds.append(pred_binary.cpu())
                all_pain_targets.append(target.cpu())
            
            # Task 2: 3-Class classification metrics
            if 'intensity_classification' in outputs and 'intensity_classification' in targets:
                pred_logits = outputs['intensity_classification']
                target = targets['intensity_classification']
                pred_class = torch.argmax(pred_logits, dim=1)
                all_intensity_preds.append(pred_class.cpu())
                all_intensity_targets.append(target.cpu())

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    # Calculate metrics for Task 1 (Binary)
    task1_acc = 0.0
    task1_f1 = 0.0
    task1_precision = 0.0
    task1_recall = 0.0
    
    if len(all_pain_preds) > 0:
        all_pain_preds = torch.cat(all_pain_preds).numpy()
        all_pain_targets = torch.cat(all_pain_targets).numpy()
        task1_acc = accuracy_score(all_pain_targets, all_pain_preds)
        task1_f1 = f1_score(all_pain_targets, all_pain_preds, zero_division=0)
        task1_precision = precision_score(all_pain_targets, all_pain_preds, zero_division=0)
        task1_recall = recall_score(all_pain_targets, all_pain_preds, zero_division=0)
    
    # Calculate metrics for Task 2 (3-Class)
    task2_acc = 0.0
    task2_f1 = 0.0
    task2_precision = 0.0
    task2_recall = 0.0
    
    if len(all_intensity_preds) > 0:
        all_intensity_preds = torch.cat(all_intensity_preds).numpy()
        all_intensity_targets = torch.cat(all_intensity_targets).numpy()
        task2_acc = accuracy_score(all_intensity_targets, all_intensity_preds)
        task2_f1 = f1_score(all_intensity_targets, all_intensity_preds, average='weighted', zero_division=0)
        task2_precision = precision_score(all_intensity_targets, all_intensity_preds, average='weighted', zero_division=0)
        task2_recall = recall_score(all_intensity_targets, all_intensity_preds, average='weighted', zero_division=0)

    return (avg_loss, 
            task1_acc, task1_f1, task1_precision, task1_recall,
            task2_acc, task2_f1, task2_precision, task2_recall)

print("✅ Training functions defined (v2.5 - Dual Classification Metrics)")

# ============================================================================
# 6. CHECKPOINT MANAGEMENT
# ============================================================================
print("\n[6] Defining checkpoint manager...")

def find_latest_checkpoint(checkpoint_dir, fold_idx):
    # Robust: choose highest epoch number, not lexicographic filename order
    candidates = []
    for p in checkpoint_dir.glob(f'checkpoint_fold_{fold_idx}_epoch_*.pt'):
        name = p.name
        try:
            epoch_str = name.split("_epoch_")[-1].replace(".pt", "")
            epoch = int(epoch_str)
            candidates.append((epoch, p))
        except Exception:
            continue
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]

def save_checkpoint(model, optimizer, scheduler, epoch, fold_idx, val_loss,
                    checkpoint_dir, is_best=False, scaler=None,
                    best_task1_f1=None, best_task2_f1=None, patience_counter=None,
                    config=None):
    checkpoint = {
        'epoch': epoch, 'fold': fold_idx, 'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(), 'val_loss': val_loss,
    }
    # Save AMP scaler for exact resume
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()
    # Save early-stopping state
    if best_task1_f1 is not None:
        checkpoint['best_task1_f1'] = float(best_task1_f1)
    if best_task2_f1 is not None:
        checkpoint['best_task2_f1'] = float(best_task2_f1)
    if patience_counter is not None:
        checkpoint['patience_counter'] = int(patience_counter)
    if config is not None:
        checkpoint['config'] = dict(config)
    # Save RNG states (helps reproducibility when resuming)
    checkpoint['rng_state'] = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }

    checkpoint_path = checkpoint_dir / f'checkpoint_fold_{fold_idx}_epoch_{epoch}.pt'
    torch.save(checkpoint, checkpoint_path)

    if is_best:
        best_path = checkpoint_dir / f'best_model_v2.5_fold_{fold_idx}.pt'
        torch.save(checkpoint, best_path)
        print(f"✅ Best model saved! (val_loss={val_loss:.4f})")

print("✅ Checkpoint manager defined")

# ============================================================================
# 7. CONFIGURATION (v2.5 - T4 GPU Optimized)
# ============================================================================
print("\n[7] Loading configuration...")

# T4 GPU has 16GB VRAM - optimize for this
if torch.cuda.is_available():
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    gpu_name = torch.cuda.get_device_name(0)
    if 'T4' in gpu_name or (gpu_memory_gb >= 14 and gpu_memory_gb < 20):
        default_batch_size = 16  # T4 optimized
        default_num_workers = 2  # T4 optimized
    elif 'L4' in gpu_name or gpu_memory_gb >= 20:
        default_batch_size = 24
        default_num_workers = 4
    elif gpu_memory_gb >= 8:
        default_batch_size = 12
        default_num_workers = 2
    else:
        default_batch_size = 8
        default_num_workers = 1
else:
    default_batch_size = 4
    default_num_workers = 0

config = {
    'batch_size': default_batch_size,
    'num_epochs': 60,
    'learning_rate': 0.0001,          # Higher than v2.4 (0.00005) - classification benefits from higher LR
                                       # v2.4 used 0.00005 for regression (conservative)
    'patience': 20,
    'min_delta': 0.001,               # For classification (F1 improvement threshold)
    'min_epochs': 5,
    'max_frames': 32,
    'weight_decay': 1e-4,
    'gradient_clip': 0.5,
    'lr_patience': 5,
    'warmup_epochs': 2,
    'num_workers': default_num_workers,
    'pin_memory': True,
    'prefetch_factor': 2,
    'use_bidirectional_lstm': False    # Standard LSTM (validated in v2.4 for small dataset)
}

transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Device: {device}")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"   GPU: {gpu_name}")
    print(f"   Memory: {gpu_memory:.1f} GB")
    print(f"   ✅ Batch size={config['batch_size']}, num_workers={config['num_workers']}")

print(f"\n✅ Config (v2.5 - Dual Classification, T4 Optimized):")
print(f"   • TASK 1: Binary Classification (Pain vs No Pain)")
print(f"   • TASK 2: 3-Class Classification (Intensity Moment)")
print(f"   • ARCHITECTURE: 2D CNN + LSTM (validated in v2.4)")
print(f"   • MOMENT WEIGHTS: M2: 4.0× (challenging), M3: 2.0× (best performer)")
print(f"   • LEARNING RATE: {config['learning_rate']}")
print(f"   • PATIENCE: {config['patience']}")
print(f"   • BATCH SIZE: {config['batch_size']} (T4 Optimized)")
print(f"   • NUM WORKERS: {config['num_workers']} (T4 Optimized)")

# ============================================================================
# 8. TRAINING LOOP
# ============================================================================
print("\n" + "="*80)
print("STARTING v2.5 TRAINING (DUAL CLASSIFICATION): 9 FOLDS")
print(f"Checkpoints will be saved to: {checkpoint_dir}")
print("="*80)

start_fold = START_FOLD
print(f"✅ Start fold: {start_fold} (set START_FOLD env var to override)")

cv_folds = splits.get('cv_folds', [])
if not cv_folds:
    cv_folds = [splits[f'fold_{i}'] for i in range(9) if f'fold_{i}' in splits]

for fold_idx, fold_data in enumerate(cv_folds[start_fold:], start=start_fold):
    if isinstance(fold_data, dict):
        train_animals = fold_data.get('train_animals', [])
        val_animals = fold_data.get('val_animals', [])
    else:
        train_animals = fold_data[0]
        val_animals = fold_data[1]

    print(f"\n{'='*80}")
    print(f"FOLD {fold_idx}/8: Val={val_animals}, Train={len(train_animals)} animals")
    print(f"{'='*80}")

    latest_checkpoint = find_latest_checkpoint(checkpoint_dir, fold_idx)
    resume = False
    if latest_checkpoint:
        print(f"\n⚠️  Found checkpoint: {latest_checkpoint.name}")
        resume = AUTO_RESUME
        print(f"   AUTO_RESUME={AUTO_RESUME} → resume={resume}")

    train_sequences = [s for s in all_sequences if s.get('animal', s.get('animal_id')) in train_animals]
    val_sequences = [s for s in all_sequences if s.get('animal', s.get('animal_id')) in val_animals]

    print(f"\n📋 Preparing data...")
    print(f"   Train: {len(train_sequences)} sequences")
    print(f"   Val: {len(val_sequences)} sequences")

    train_dataset = FacialPainDataset_v2_5(train_sequences, sequence_dir,
                                           max_frames=config['max_frames'],
                                           transform=transform, augment=True)
    val_dataset = FacialPainDataset_v2_5(val_sequences, sequence_dir,
                                         max_frames=config['max_frames'],
                                         transform=transform, augment=False)

    # Calculate class weights for class imbalance
    # Task 1: Binary classification
    train_pain_count = sum(1 for s in train_sequences if s.get('moment') in ['M2', 'M3', 'M4'])
    train_no_pain_count = sum(1 for s in train_sequences if s.get('moment') in ['M0', 'M1'])
    if train_pain_count > 0 and train_no_pain_count > 0:
        pos_weight = train_no_pain_count / train_pain_count
        print(f"   📊 Task 1 - Class balance: Pain={train_pain_count}, No Pain={train_no_pain_count}")
        print(f"   📊 Task 1 - Positive class weight: {pos_weight:.2f}")
    else:
        pos_weight = None
    
    # Task 2: 3-Class classification
    train_no_pain_count = sum(1 for s in train_sequences if s.get('moment') in ['M0', 'M1'])
    train_acute_count = sum(1 for s in train_sequences if s.get('moment') == 'M2')
    train_residual_count = sum(1 for s in train_sequences if s.get('moment') in ['M3', 'M4'])
    total_3class = train_no_pain_count + train_acute_count + train_residual_count
    
    if total_3class > 0 and train_no_pain_count > 0 and train_acute_count > 0 and train_residual_count > 0:
        # Inverse frequency weighting
        intensity_class_weights = [
            total_3class / (3 * train_no_pain_count),      # No Pain
            total_3class / (3 * train_acute_count),        # Acute Pain
            total_3class / (3 * train_residual_count)     # Residual Pain
        ]
        print(f"   📊 Task 2 - Class balance: No Pain={train_no_pain_count}, Acute={train_acute_count}, Residual={train_residual_count}")
        print(f"   📊 Task 2 - Class weights: {[f'{w:.2f}' for w in intensity_class_weights]}")
    else:
        intensity_class_weights = None

    print(f"   Creating stratified sampler...")
    train_sampler = create_stratified_sampler(train_dataset)
    num_workers = config.get('num_workers', 2)
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], sampler=train_sampler,
        num_workers=num_workers, pin_memory=config.get('pin_memory', True),
        persistent_workers=(num_workers > 0), prefetch_factor=config.get('prefetch_factor', 2)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=False,
        num_workers=num_workers, pin_memory=config.get('pin_memory', True),
        persistent_workers=(num_workers > 0), prefetch_factor=config.get('prefetch_factor', 2)
    )

    print(f"✅ Data preparation complete!")

    print(f"\n   Initializing model (2D CNN + LSTM, Dual Classification)...")
    model = TemporalPainModel_v2_5(
        num_frames=config['max_frames'],
        lstm_hidden_size=128,
        use_bidirectional=config.get('use_bidirectional_lstm', False)
    ).to(device)
    
    criterion = WeightedDualClassificationLoss(
        pos_weight=pos_weight,
        intensity_class_weights=intensity_class_weights,
        task1_weight=1.0,
        task2_weight=1.0
    )

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config['learning_rate'],
                                  weight_decay=config['weight_decay'])

    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5,
        patience=config.get('lr_patience', 5), min_lr=1e-7
    )

    use_amp = torch.cuda.is_available() and hasattr(torch.cuda, 'amp')
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        print(f"   ✅ Mixed precision training enabled")

    start_epoch = 0
    best_val_loss = float('inf')
    best_task1_f1 = 0.0
    best_task2_f1 = 0.0
    patience_counter = 0
    min_delta = config.get('min_delta', 0.001)
    min_epochs = config.get('min_epochs', 5)
    warmup_epochs = config.get('warmup_epochs', 2)

    if resume and latest_checkpoint:
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        best_task1_f1 = checkpoint.get('best_task1_f1', best_task1_f1)
        best_task2_f1 = checkpoint.get('best_task2_f1', best_task2_f1)
        patience_counter = checkpoint.get('patience_counter', patience_counter)
        if scaler is not None and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        # Restore RNG states if present
        rng = checkpoint.get('rng_state')
        if rng is not None:
            try:
                random.setstate(rng.get('python'))
                np.random.set_state(rng.get('numpy'))
                torch.set_rng_state(rng.get('torch'))
                if torch.cuda.is_available() and rng.get('torch_cuda') is not None:
                    torch.cuda.set_rng_state_all(rng.get('torch_cuda'))
            except Exception:
                pass
        print(f"✅ Resumed from epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, config['num_epochs']):
        print(f"\n📊 Epoch {epoch+1}/{config['num_epochs']}")

        if epoch < warmup_epochs:
            base_lr = config['learning_rate']
            warmup_lr = base_lr * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
            print(f"   🔥 Warmup LR: {warmup_lr:.7f}")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, config, scaler)
        val_results = validate(model, val_loader, criterion, device)
        (val_loss, 
         task1_acc, task1_f1, task1_precision, task1_recall,
         task2_acc, task2_f1, task2_precision, task2_recall) = val_results

        if epoch >= warmup_epochs:
            scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"   Train: {train_loss:.4f}, Val: {val_loss:.4f}")
        print(f"   Task 1 (Binary): Acc={task1_acc:.3f}, F1={task1_f1:.3f}, P={task1_precision:.3f}, R={task1_recall:.3f}")
        print(f"   Task 2 (3-Class): Acc={task2_acc:.3f}, F1={task2_f1:.3f}, P={task2_precision:.3f}, R={task2_recall:.3f}")
        print(f"   LR: {current_lr:.7f}")

        # Use combined F1 score for best model selection (average of both tasks)
        combined_f1 = (task1_f1 + task2_f1) / 2.0
        best_combined_f1 = (best_task1_f1 + best_task2_f1) / 2.0
        improvement = combined_f1 - best_combined_f1
        is_best = improvement > min_delta

        if is_best:
            best_val_loss = val_loss
            best_task1_f1 = task1_f1
            best_task2_f1 = task2_f1
            patience_counter = 0
            save_checkpoint(model, optimizer, scheduler, epoch, fold_idx, val_loss,
                            checkpoint_dir, is_best=True, scaler=scaler,
                            best_task1_f1=best_task1_f1, best_task2_f1=best_task2_f1,
                            patience_counter=patience_counter, config=config)
        else:
            patience_counter += 1
            print(f"   ⏳ Patience: {patience_counter}/{config['patience']} (Combined F1 improvement: {improvement:.4f})")

        save_checkpoint(model, optimizer, scheduler, epoch, fold_idx, val_loss,
                        checkpoint_dir, is_best=False, scaler=scaler,
                        best_task1_f1=best_task1_f1, best_task2_f1=best_task2_f1,
                        patience_counter=patience_counter, config=config)

        if epoch >= min_epochs and patience_counter >= config['patience']:
            print(f"   🛑 Early stopping triggered! (trained for {epoch+1} epochs)")
            break

    print(f"✅ Fold {fold_idx} complete! Best Val Loss: {best_val_loss:.4f}")
    print(f"   Best Task 1 F1: {best_task1_f1:.3f}, Best Task 2 F1: {best_task2_f1:.3f}")

    # Cleanup old checkpoints (keep last 3)
    checkpoints = sorted(checkpoint_dir.glob(f'checkpoint_fold_{fold_idx}_epoch_*.pt'))
    for old_checkpoint in checkpoints[:-3]:
        old_checkpoint.unlink()

print(f"\n{'='*80}")
print(f"🎉 ALL v2.5 TRAINING COMPLETE!")
print(f"{'='*80}")
print(f"\n📁 Models saved in: {checkpoint_dir}")
print(f"   Best models: best_model_v2.5_fold_*.pt")
print(f"\n📊 Key Features:")
print(f"   • Dual classification tasks:")
print(f"     - Task 1: Binary (Pain vs No Pain)")
print(f"     - Task 2: 3-Class (Intensity Moment: No Pain, Acute Pain, Residual Pain)")
print(f"   • 2D CNN + LSTM architecture (validated in v2.4)")
print(f"   • T4 GPU optimized (batch_size={config['batch_size']}, num_workers={config['num_workers']})")
print(f"   • Moment-weighted loss based on v2.4 insights (M2: 4.0×, M3: 2.0×)")
print(f"   • Comprehensive metrics for both tasks (Accuracy, F1, Precision, Recall)")
print(f"\n📈 v2.4 Insights Incorporated:")
print(f"   • M2 (peak pain): CRITICAL but challenging (r=-0.5715 in test)")
print(f"   • M3 (declining): BEST performer (R²=0.4209, r=0.9620)")
print(f"   • M1 performs BETTER than M0 (MAE: 1.300 vs 1.909)")
print(f"   • Moment weights: M2=4.0 (reduced from 10.0), M3=2.0")
print(f"   • Architecture: 2D CNN + LSTM validated (test R²=0.3125)")
print(f"   • Best folds for ensemble: 2, 7, 6 (test R²: 0.5322, 0.4158, 0.3255)")

