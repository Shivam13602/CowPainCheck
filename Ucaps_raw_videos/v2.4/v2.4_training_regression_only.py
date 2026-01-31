# ============================================================================
# PHASE 2: MODEL TRAINING (v2.4 - REGRESSION ONLY WITH DATA-DRIVEN WEIGHTS)
#
# Key Features:
# - REGRESSION ONLY: Predicts 7 facial features + Total Facial Scale
# - DATA-DRIVEN WEIGHTS: Based on comprehensive analysis (dataanlasis.md)
# - 2D CNN + LSTM: More suitable for small datasets than 3D CNN
# - MOMENT WEIGHTS: M0:1.0, M1:1.0, M2:4.0, M3:2.0, M4:1.2 (data-driven)
# - FEATURE WEIGHTS: Based on correlation + test performance analysis
#
# v2.4 Improvements (Based on v2.3):
# - Removed classification task (regression only)
# - Switched from 3D CNN to 2D CNN + LSTM (better for small datasets)
# - Updated moment weights based on test performance analysis
# - Updated feature weights based on comprehensive correlation analysis
# - Standard LSTM (not bidirectional) to reduce overfitting on small dataset
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.transforms as transforms

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
# SETUP: Mount Drive and Verify Paths
# ============================================================================
print("="*80)
print("PHASE 2: MODEL TRAINING (v2.4 - REGRESSION ONLY WITH DATA-DRIVEN WEIGHTS)")
print("="*80)
print("Architecture: 2D CNN + LSTM (optimized for small dataset)")
print("Weights: Data-driven based on comprehensive analysis")
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
checkpoint_dir = project_dir / 'checkpoints_v2.4'
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
# 1. MODEL ARCHITECTURE (v2.4 - 2D CNN + LSTM)
# ============================================================================
print("\n[1] Defining model architecture (2D CNN + LSTM)...")

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


class TemporalPainModel_v2_4(nn.Module):
    """
    v2.4: 2D CNN + LSTM architecture (optimized for small datasets)
    
    Architecture:
    - 2D CNN: Extracts spatial features from each frame individually
    - LSTM: Processes sequence of frame features (standard, not bidirectional)
    - Attention: Aggregates temporal information
    - Output: 7 facial features + Total Facial Scale (calculated)
    
    Rationale for 2D CNN + LSTM:
    - More parameter-efficient than 3D CNN
    - Better suited for small datasets
    - Standard LSTM (not bidirectional) reduces overfitting risk
    """
    def __init__(self, num_frames=32, lstm_hidden_size=128, use_bidirectional=False):
        super(TemporalPainModel_v2_4, self).__init__()
        
        # 2D CNN for spatial feature extraction
        # Input: (batch, frames, 3, H, W) -> process each frame
        # Using ResNet-like architecture for better feature extraction
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
        
        # Regression output heads
        self.output_heads = nn.ModuleDict({
            'Orbital_tightening': nn.Linear(self.lstm_output_size, 1),
            'Tension_above_eyes': nn.Linear(self.lstm_output_size, 1),
            'Cheek_tightening': nn.Linear(self.lstm_output_size, 1),
            'Ears_frontal': nn.Linear(self.lstm_output_size, 1),
            'Ears_lateral': nn.Linear(self.lstm_output_size, 1),
            'Lip_jaw_profile': nn.Linear(self.lstm_output_size, 1),
            'Nostril_muzzle': nn.Linear(self.lstm_output_size, 1),
        })
        
        # Optional: Direct prediction of Total (for consistency loss)
        self.total_head = nn.Linear(self.lstm_output_size, 1)
        
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
        
        # Regression outputs
        outputs = {}
        for task, head in self.output_heads.items():
            outputs[task] = head(context).squeeze(-1)
        
        # Calculate Total Facial Scale from 7 features (UNESP-Botucatu validated method)
        individual_features = [
            outputs['Orbital_tightening'],
            outputs['Tension_above_eyes'],
            outputs['Cheek_tightening'],
            outputs['Ears_frontal'],
            outputs['Ears_lateral'],
            outputs['Lip_jaw_profile'],
            outputs['Nostril_muzzle']
        ]
        outputs['Total_Facial_scale_calculated'] = torch.stack(individual_features, dim=0).sum(dim=0)
        
        # Also predict Total directly (for consistency loss)
        outputs['Total_Facial_scale_predicted'] = self.total_head(context).squeeze(-1)
        
        # Use calculated Total as the main output (validated method)
        outputs['Total_Facial_scale'] = outputs['Total_Facial_scale_calculated']
        
        return outputs, attention_weights

print("✅ Model defined (v2.4 - 2D CNN + LSTM)")

# ============================================================================
# 2. DATASET CLASS (v2.4 - Regression Only)
# ============================================================================
print("\n[2] Defining dataset class...")

class FacialPainDataset_v2_4(Dataset):
    """v2.4: Regression only - no classification labels"""
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

        self.csv_label_cols = [
            'Orbital_tightening', 'Tension_above_eyes', 'Cheek_tightening',
            'Ears_frontal', 'Ears_lateral', 'Lip_jaw_profile', 'Nostril_muzzle',
            'Total.Facial.scale'
        ]
        self.label_name_map = {
            'Orbital_tightening': 'Orbital_tightening',
            'Tension_above_eyes': 'Tension_above_eyes',
            'Cheek_tightening': 'Cheek_tightening',
            'Ears_frontal': 'Ears_frontal',
            'Ears_lateral': 'Ears_lateral',
            'Lip_jaw_profile': 'Lip_jaw_profile',
            'Nostril_muzzle': 'Nostril_muzzle',
            'Total.Facial.scale': 'Total_Facial_scale',
        }

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

            labels = {}
            for csv_col in self.csv_label_cols:
                model_col = self.label_name_map.get(csv_col, csv_col.replace('.', '_'))
                labels[model_col] = torch.tensor(0.0, dtype=torch.float32)

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

        labels = {}
        for csv_col in self.csv_label_cols:
            val = seq_info.get(csv_col, np.nan)
            if pd.isna(val):
                model_col = self.label_name_map.get(csv_col, csv_col.replace('.', '_'))
                val = seq_info.get(model_col, np.nan)
            model_col = self.label_name_map.get(csv_col, csv_col.replace('.', '_'))
            labels[model_col] = torch.tensor(val if not pd.isna(val) else 0.0, dtype=torch.float32)

        metadata = {
            'animal': seq_info.get('animal', seq_info.get('animal_id', 'unknown')),
            'moment': seq_info.get('moment', 'unknown'),
            'sequence_id': seq_info.get('sequence_id', f'seq_{idx}')
        }

        return frames, labels, metadata

print("✅ Dataset class defined (v2.4 - Regression Only)")

# ============================================================================
# 3. WEIGHTED LOSS FUNCTION (v2.4 - Data-Driven Weights)
# ============================================================================
print("\n[3] Defining weighted loss function with data-driven weights...")

class WeightedPainLoss(nn.Module):
    """
    v2.4: Regression loss with data-driven moment and feature weights
    
    Weights based on comprehensive analysis (dataanlasis.md):
    - Moment weights: Data-driven from test performance analysis
    - Feature weights: Based on correlation + test performance
    """
    def __init__(self, moment_weights=None, feature_weights=None, normalize_features=True,
                 consistency_weight=0.1):
        super(WeightedPainLoss, self).__init__()

        # Data-driven moment weights (from test performance analysis)
        self.moment_weights = moment_weights or {
            'M0': 1.0,   # Baseline - good performance (MAE=1.909)
            'M1': 1.0,   # REDUCED from 2.0 - performs BETTER than M0 (MAE=1.300)
            'M2': 4.0,   # CRITICAL - REDUCED from 10.0 (was causing over-fitting)
            'M3': 2.0,   # Declining pain (MAE=2.412)
            'M4': 1.2    # Recovery assessment (MAE=1.608)
        }
        
        # Feature weights based on comprehensive correlation analysis (from dataanlasis.md)
        # Using exact normalized weights from analysis
        if feature_weights is None:
            # Direct normalized weights from comprehensive analysis
            self.feature_weights = {
                'Total_Facial_scale': 2.0,         # Highest correlation (r=0.843) - primary target
                'Orbital_tightening': 1.9,         # High correlation (r=0.689) + best test R² (0.199)
                'Ears_lateral': 1.7,               # Very high correlation (r=0.697), moderate test
                'Lip_jaw_profile': 1.6,            # Very high correlation (r=0.695) but negative test R²
                'Ears_frontal': 1.5,               # High correlation (r=0.650) but negative test R²
                'Cheek_tightening': 1.4,           # High correlation (r=0.644), moderate test
                'Tension_above_eyes': 1.1,         # Moderate correlation (r=0.526)
                'Nostril_muzzle': 1.1              # Moderate correlation (r=0.511)
            }
        else:
            self.feature_weights = feature_weights

        self.consistency_weight = consistency_weight

        print("\n" + "="*70)
        print("WEIGHTED LOSS CONFIGURATION (v2.4 - Data-Driven)")
        print("="*70)
        print("\n⏰ Moment Weights (Data-Driven from Test Performance):")
        for moment, weight in sorted(self.moment_weights.items(), key=lambda x: x[1], reverse=True):
            print(f"   {moment:25s}: {weight:.1f}×")
        print("\n🔗 Feature Weights (Based on Correlation + Test Performance):")
        for feature, weight in sorted(self.feature_weights.items(), key=lambda x: x[1], reverse=True):
            print(f"   {feature:25s}: {weight:.2f}×")
        print(f"\n🔗 Consistency Loss Weight: {self.consistency_weight}")
        print("="*70 + "\n")

    def forward(self, predictions, targets, moments):
        total_loss = 0.0
        num_tasks = 0.0

        # Regression loss for individual features
        for task in predictions.keys():
            if task in ['Total_Facial_scale_calculated', 'Total_Facial_scale_predicted',
                        'Total_Facial_scale']:
                continue
            if task not in targets:
                continue

            pred = predictions[task]
            target = targets[task]
            sample_losses = (pred - target) ** 2

            moment_weight_tensor = torch.tensor(
                [self.moment_weights.get(m, 1.0) for m in moments],
                device=pred.device, dtype=torch.float32
            )
            feature_weight = self.feature_weights.get(task, 1.0)
            combined_weights = moment_weight_tensor * feature_weight

            total_loss += (sample_losses * combined_weights).mean()
            num_tasks += 1

        # Total Scale - Calculated
        if 'Total_Facial_scale_calculated' in predictions and 'Total_Facial_scale' in targets:
            total_calc = predictions['Total_Facial_scale_calculated']
            target_total = targets['Total_Facial_scale']
            sample_losses_calc = (total_calc - target_total) ** 2
            moment_weight_tensor = torch.tensor(
                [self.moment_weights.get(m, 1.0) for m in moments],
                device=total_calc.device, dtype=torch.float32
            )
            feature_weight_total = self.feature_weights.get('Total_Facial_scale', 1.0)
            combined_weights = moment_weight_tensor * feature_weight_total
            loss_calculated = (sample_losses_calc * combined_weights).mean()

            total_loss += loss_calculated * 0.75
            num_tasks += 0.75

        # Total Scale - Predicted (for consistency)
        if 'Total_Facial_scale_predicted' in predictions and 'Total_Facial_scale' in targets:
            total_pred = predictions['Total_Facial_scale_predicted']
            target_total = targets['Total_Facial_scale']
            sample_losses_pred = (total_pred - target_total) ** 2
            moment_weight_tensor = torch.tensor(
                [self.moment_weights.get(m, 1.0) for m in moments],
                device=total_pred.device, dtype=torch.float32
            )
            feature_weight_total = self.feature_weights.get('Total_Facial_scale', 1.0)
            combined_weights = moment_weight_tensor * feature_weight_total
            loss_predicted = (sample_losses_pred * combined_weights).mean()

            total_loss += loss_predicted * 0.75
            num_tasks += 0.75

        # Consistency Loss
        if 'Total_Facial_scale_predicted' in predictions and 'Total_Facial_scale_calculated' in predictions:
            total_pred = predictions['Total_Facial_scale_predicted']
            total_calc = predictions['Total_Facial_scale_calculated']
            consistency_loss = F.mse_loss(total_pred, total_calc)
            total_loss += self.consistency_weight * consistency_loss

        # Average the loss
        avg_loss = total_loss / num_tasks if num_tasks > 0 else 0.0

        return avg_loss

print("✅ Weighted loss function defined (v2.4)")

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
# 5. TRAINING FUNCTIONS
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

    with torch.no_grad():
        for frames, labels, metadata in dataloader:
            frames = frames.to(device, non_blocking=True)
            moments = metadata['moment']
            targets = {k: v.to(device, non_blocking=True) for k, v in labels.items()}

            outputs, _ = model(frames)
            loss = criterion(outputs, targets, moments)

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss

print("✅ Training functions defined")

# ============================================================================
# 6. CHECKPOINT MANAGEMENT
# ============================================================================
print("\n[6] Defining checkpoint manager...")

def find_latest_checkpoint(checkpoint_dir, fold_idx):
    checkpoints = sorted(checkpoint_dir.glob(f'checkpoint_fold_{fold_idx}_epoch_*.pt'))
    if checkpoints:
        return checkpoints[-1]
    return None

def save_checkpoint(model, optimizer, scheduler, epoch, fold_idx, val_loss,
                    checkpoint_dir, is_best=False):
    checkpoint = {
        'epoch': epoch, 'fold': fold_idx, 'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(), 'val_loss': val_loss,
    }

    checkpoint_path = checkpoint_dir / f'checkpoint_fold_{fold_idx}_epoch_{epoch}.pt'
    torch.save(checkpoint, checkpoint_path)

    if is_best:
        best_path = checkpoint_dir / f'best_model_v2.4_fold_{fold_idx}.pt'
        torch.save(checkpoint, best_path)
        print(f"✅ Best model saved! (val_loss={val_loss:.4f})")

print("✅ Checkpoint manager defined")

# ============================================================================
# 7. CONFIGURATION (v2.4 - Optimized for Small Dataset)
# ============================================================================
print("\n[7] Loading configuration...")

if torch.cuda.is_available():
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    gpu_name = torch.cuda.get_device_name(0)
    if 'L4' in gpu_name or gpu_memory_gb >= 20:
        default_batch_size = 32  # Reduced for 2D CNN (more memory per sample)
    elif gpu_memory_gb >= 15:
        default_batch_size = 24
    else:
        default_batch_size = 16
else:
    default_batch_size = 8

config = {
    'batch_size': default_batch_size,
    'num_epochs': 60,
    'learning_rate': 0.00005,          # Conservative LR for small dataset
    'patience': 20,                    # Increased patience
    'min_delta': 0.0005,
    'min_epochs': 5,
    'max_frames': 32,
    'weight_decay': 1e-4,
    'gradient_clip': 0.5,
    'lr_patience': 5,
    'warmup_epochs': 2,
    'num_workers': 4,  # Increased for Colab L4 (can handle more workers)
    'pin_memory': True,
    'prefetch_factor': 4,  # Increased for better data pipeline throughput
    'use_bidirectional_lstm': False    # Standard LSTM for small dataset
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

print(f"\n✅ Config (v2.4 - Regression Only, Data-Driven Weights):")
print(f"   • ARCHITECTURE: 2D CNN + LSTM (standard, not bidirectional)")
print(f"   • TASK: Regression only (7 features + Total)")
print(f"   • MOMENT WEIGHTS: Data-driven (M2: 4.0×, reduced from 10.0×)")
print(f"   • FEATURE WEIGHTS: Based on correlation + test performance")
print(f"   • LEARNING RATE: {config['learning_rate']}")
print(f"   • PATIENCE: {config['patience']}")

# ============================================================================
# 8. TRAINING LOOP
# ============================================================================
print("\n" + "="*80)
print("STARTING v2.4 TRAINING (REGRESSION ONLY): 9 FOLDS")
print(f"Checkpoints will be saved to: {checkpoint_dir}")
print("="*80)

resume_fold = input("Start from fold (0-8) or press Enter to start from 0: ").strip()
start_fold = int(resume_fold) if resume_fold else 0

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
        resume = input("Resume from checkpoint? (y/n): ").lower() == 'y'

    train_sequences = [s for s in all_sequences if s.get('animal', s.get('animal_id')) in train_animals]
    val_sequences = [s for s in all_sequences if s.get('animal', s.get('animal_id')) in val_animals]

    print(f"\n📋 Preparing data...")
    print(f"   Train: {len(train_sequences)} sequences")
    print(f"   Val: {len(val_sequences)} sequences")

    train_dataset = FacialPainDataset_v2_4(train_sequences, sequence_dir,
                                           max_frames=config['max_frames'],
                                           transform=transform, augment=True)
    val_dataset = FacialPainDataset_v2_4(val_sequences, sequence_dir,
                                         max_frames=config['max_frames'],
                                         transform=transform, augment=False)

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

    print(f"\n   Initializing model (2D CNN + LSTM)...")
    model = TemporalPainModel_v2_4(
        num_frames=config['max_frames'],
        lstm_hidden_size=128,
        use_bidirectional=config.get('use_bidirectional_lstm', False)
    ).to(device)
    
    criterion = WeightedPainLoss()  # Uses data-driven weights by default

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
    patience_counter = 0
    min_delta = config.get('min_delta', 0.0001)
    min_epochs = config.get('min_epochs', 5)
    warmup_epochs = config.get('warmup_epochs', 2)

    if resume and latest_checkpoint:
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('val_loss', float('inf'))
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
        val_loss = validate(model, val_loader, criterion, device)

        if epoch >= warmup_epochs:
            scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"   Train: {train_loss:.4f}, Val: {val_loss:.4f}, LR: {current_lr:.7f}")

        improvement = best_val_loss - val_loss
        is_best = improvement > min_delta

        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model, optimizer, scheduler, epoch, fold_idx, val_loss,
                            checkpoint_dir, is_best=True)
        else:
            patience_counter += 1
            print(f"   ⏳ Patience: {patience_counter}/{config['patience']} (improvement: {improvement:.4f})")

        save_checkpoint(model, optimizer, scheduler, epoch, fold_idx, val_loss, checkpoint_dir, is_best=False)

        if epoch >= min_epochs and patience_counter >= config['patience']:
            print(f"   🛑 Early stopping triggered! (trained for {epoch+1} epochs)")
            break

    print(f"✅ Fold {fold_idx} complete! Best: {best_val_loss:.4f}")

    # Cleanup old checkpoints (keep last 3)
    checkpoints = sorted(checkpoint_dir.glob(f'checkpoint_fold_{fold_idx}_epoch_*.pt'))
    for old_checkpoint in checkpoints[:-3]:
        old_checkpoint.unlink()

print(f"\n{'='*80}")
print(f"🎉 ALL v2.4 TRAINING COMPLETE!")
print(f"{'='*80}")
print(f"\n📁 Models saved in: {checkpoint_dir}")
print(f"   Best models: best_model_v2.4_fold_*.pt")
print(f"\n📊 Key Changes from v2.3:")
print(f"   • Removed classification task")
print(f"   • Switched to 2D CNN + LSTM architecture")
print(f"   • Updated moment weights (M2: 10.0 → 4.0)")
print(f"   • Updated feature weights (data-driven)")

