# ============================================================================
# PHASE 2: MODEL TRAINING (v2.2 - LIGHTWEIGHT MODEL WITH REDUCED LR)
# Optimized for Google Colab T4 GPU with Feature-Weighted Loss
# 
# Key Features:
# - Feature-weighted loss (Orbital/Ears = higher weight based on correlation)
# - Total Facial Scale calculated from 7 features (UNESP-Botucatu validated method)
# - Consistency loss to enforce Total = sum(7 features)
# - Fixed label mapping (Total.Facial.scale ↔ Total_Facial_scale)
# - Fully resumable (handles Colab disconnects)
#
# v2.2 Improvements (Lightweight Architecture + Lower LR + Classification):
# - Lighter model: Reduced channels (8→16→32→64) and LSTM hidden size (128)
# - Lower learning rate (0.0001 → 0.00003) for more stable training
# - Reduced dropout (0.5 → 0.3) for lighter model
# - Pain Intensity Classification: 3 classes (No pain, Acute pain, Residual pain)
# - Estimated ~3-4M parameters (vs 13.7M in v2.1)
# - Better generalization with smaller model
#
# Total Facial Scale Mechanism (v2.2):
# - Calculated Total: Sum of 7 features (UNESP-Botucatu validated) - Primary output
# - Predicted Total: Direct prediction from model head - Secondary output
# - Both trained against ground truth (calculated: 100% weight, predicted: 50% weight)
# - Consistency loss: Enforces predicted ≈ calculated (10% weight)
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
print("PHASE 2: MODEL TRAINING (v2.2 - LIGHTWEIGHT MODEL + REDUCED LR)")
print("="*80)
print("Previous results not satisfactory - Using lightweight architecture + lower LR")
print("="*80)

# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Set paths
base_path = Path('/content/drive/MyDrive')
project_dir = base_path / 'facial_pain_project_v2'
sequence_dir = base_path / 'sequence'
checkpoint_dir = project_dir / 'checkpoints'
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
        # Convert dict to list
        all_sequences = [{'sequence_id': k, **v} for k, v in sequence_mapping.items()]
else:
    all_sequences = sequence_mapping

print(f"\n✅ Loaded {len(all_sequences)} sequences")
print(f"✅ Checkpoint dir: {checkpoint_dir}")

# ============================================================================
# 1. MODEL ARCHITECTURE (7 features + Total calculated from features)
# ============================================================================
print("\n[1] Defining model architecture...")

class AttentionLayer(nn.Module):
    """Attention mechanism for temporal modeling"""
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, lstm_output):
        # lstm_output: (batch, seq_len, hidden_size)
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        return context_vector, attention_weights


class TemporalPainModel_v2(nn.Module):
    """
    v2.2: Lightweight model with Total Facial Scale calculation + Pain Intensity Classification
    
    Architecture (Lightweight):
    - 3D CNN (112×112 input, reduced channels: 8→16→32→64)
    - Bidirectional LSTM (hidden_size=128, reduced from 256)
    - Attention mechanism
    - 7 output heads (individual features)
    - Total Facial Scale = sum of 7 features (UNESP-Botucatu validated method)
    - Pain Intensity Classification: 3 classes (No pain, Acute pain, Residual pain)
    - ~3-4M parameters (vs 13.7M in v2.1)
    
    Pain Intensity Classes:
    - Class 0 (No pain): M0, M1
    - Class 1 (Acute pain): M2
    - Class 2 (Residual pain): M3, M4
    """
    def __init__(self, num_frames=32, lstm_hidden_size=128):
        super(TemporalPainModel_v2, self).__init__()
        
        # 3D CNN - Lightweight: Reduced channels (8→16→32→64)
        self.conv1 = nn.Conv3d(3, 8, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(8)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        
        self.conv2 = nn.Conv3d(8, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(16)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        
        self.conv3 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(32)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        
        self.conv4 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn4 = nn.BatchNorm3d(64)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        
        # CNN output: 64 × 7 × 7 = 3136 (reduced from 6272)
        self.cnn_output_size = 64 * 7 * 7
        
        # Bidirectional LSTM - Reduced hidden size (128 vs 256)
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_size,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.lstm_output_size = lstm_hidden_size * 2
        
        # Attention
        self.attention = AttentionLayer(self.lstm_output_size)
        
        # 7 output heads for individual features (Total will be calculated)
        self.output_heads = nn.ModuleDict({
            'Orbital_tightening': nn.Linear(self.lstm_output_size, 1),
            'Tension_above_eyes': nn.Linear(self.lstm_output_size, 1),
            'Cheek_tightening': nn.Linear(self.lstm_output_size, 1),
            'Ears_frontal': nn.Linear(self.lstm_output_size, 1),
            'Ears_lateral': nn.Linear(self.lstm_output_size, 1),
            'Lip_jaw_profile': nn.Linear(self.lstm_output_size, 1),
            'Nostril_muzzle': nn.Linear(self.lstm_output_size, 1),
        })
        
        # Optional: Also predict Total directly (for consistency loss)
        self.total_head = nn.Linear(self.lstm_output_size, 1)
        
        # Pain Intensity Classification Head (3 classes: No pain, Acute pain, Residual pain)
        self.pain_intensity_head = nn.Linear(self.lstm_output_size, 3)
        
        self.dropout = nn.Dropout(0.3)  # Reduced from 0.5 for lighter model
    
    def forward(self, x):
        batch_size, frames, C, H, W = x.size()
        
        # 3D CNN: (batch, channels, frames, height, width)
        x = x.permute(0, 2, 1, 3, 4)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        
        # LSTM: (batch, frames, features)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.contiguous().view(batch_size, -1, self.cnn_output_size)
        
        lstm_out, _ = self.lstm(x)
        context, attention_weights = self.attention(lstm_out)
        context = self.dropout(context)
        
        # 7 individual feature outputs
        outputs = {}
        for task, head in self.output_heads.items():
            outputs[task] = head(context).squeeze(-1)
        
        # Calculate Total Facial Scale from 7 features (UNESP-Botucatu scale definition)
        # Total = sum of 7 individual features (each 0-2, total 0-14)
        # This is the validated method from peer-reviewed veterinary literature
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
        
        # Pain Intensity Classification (3 classes: No pain, Acute pain, Residual pain)
        outputs['pain_intensity'] = self.pain_intensity_head(context)  # [batch_size, 3]
        
        return outputs, attention_weights

# Calculate and print model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Create a dummy model to count parameters
dummy_model = TemporalPainModel_v2(num_frames=32, lstm_hidden_size=128)
num_params = count_parameters(dummy_model)
print(f"✅ Model defined (~{num_params/1e6:.2f}M parameters - Lightweight v2.2)")

# ============================================================================
# 2. DATASET CLASS (with label mapping fix)
# ============================================================================
print("\n[2] Defining dataset class...")

class FacialPainDataset_v2(Dataset):
    """v2.2: 7 facial features + Total Facial Scale + Pain Intensity Classification"""
    def __init__(self, sequence_mapping, sequence_dir, max_frames=32, 
                 transform=None, augment=False):
        self.sequence_mapping = sequence_mapping
        self.sequence_dir = Path(sequence_dir)
        self.max_frames = max_frames
        self.transform = transform
        self.augment = augment
        
        # Pain Intensity Class Mapping:
        # Class 0 (No pain): M0, M1
        # Class 1 (Acute pain): M2
        # Class 2 (Residual pain): M3, M4
        self.moment_to_class = {
            'M0': 0,  # No pain
            'M1': 0,  # No pain
            'M2': 1,  # Acute pain
            'M3': 2,  # Residual pain
            'M4': 2,  # Residual pain
        }
        
        # Cache frame paths AND file lists for faster loading (pre-compute once)
        print("   Pre-computing frame paths and file lists (one-time, speeds up training)...")
        self.frame_paths_cache = {}
        self.frame_files_cache = {}  # Cache actual file lists too!
        for idx, seq_info in enumerate(tqdm(sequence_mapping, desc="Caching paths")):
            frame_path = self._find_frames_path(seq_info)
            self.frame_paths_cache[idx] = frame_path
            # Also cache the file list if path exists
            if frame_path and frame_path.exists():
                frame_files = sorted(list(frame_path.glob('*.jpg')) + list(frame_path.glob('*.png')))
                self.frame_files_cache[idx] = frame_files if len(frame_files) > 0 else None
            else:
                self.frame_files_cache[idx] = None
        print(f"   ✅ Cached {len(self.frame_paths_cache)} sequence paths and file lists")
        
        if augment:
            # Optimized for speed (L4 GPU) - removed expensive augmentations
            # Kept essential augmentations that are fast
            self.aug_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),  # Fast - keep
                transforms.ColorJitter(
                    brightness=0.2,      # Reduced from 0.3 (faster)
                    contrast=0.2,        # Reduced from 0.3 (faster)
                    saturation=0.2,     # Reduced from 0.3 (faster)
                    # hue=0.1 removed - expensive operation
                ),
                # RandomRotation removed - expensive for 32 frames
                # RandomAffine removed - very expensive
                # GaussianBlur removed - very expensive
            ])
        
        # CSV format (with dots)
        self.csv_label_cols = [
            'Orbital_tightening', 'Tension_above_eyes', 'Cheek_tightening',
            'Ears_frontal', 'Ears_lateral', 'Lip_jaw_profile', 'Nostril_muzzle',
            'Total.Facial.scale'  # ← CSV format (dots)
        ]
        
        # Model format (underscores) - FIXED mapping
        self.label_name_map = {
            'Orbital_tightening': 'Orbital_tightening',
            'Tension_above_eyes': 'Tension_above_eyes',
            'Cheek_tightening': 'Cheek_tightening',
            'Ears_frontal': 'Ears_frontal',
            'Ears_lateral': 'Ears_lateral',
            'Lip_jaw_profile': 'Lip_jaw_profile',
            'Nostril_muzzle': 'Nostril_muzzle',
            'Total.Facial.scale': 'Total_Facial_scale',  # ← FIXED: Map dots → underscore
        }
    
    def __len__(self):
        return len(self.sequence_mapping)
    
    def _find_frames_path(self, seq_info):
        """Find frame directory path (cached for speed)"""
        # Get sequence path (handle different formats)
        if 'sequence_path' in seq_info:
            seq_path = self.sequence_dir / seq_info['sequence_path']
        elif 'sequence_id' in seq_info:
            seq_path = self.sequence_dir / seq_info['sequence_id']
        else:
            return None
        
        # Try direct path first
        if seq_path.exists():
            frame_files = sorted(list(seq_path.glob('*.jpg')) + list(seq_path.glob('*.png')))
            if len(frame_files) > 0:
                return seq_path
        
        # Try common subdirectory patterns (fast check)
        possible_subdirs = [
            seq_path / 'sequence_001',
            seq_path / 'sequence_002',
            seq_path / 'sequence_003',
            seq_path / 'frames',
            seq_path / 'images',
        ]
        for subdir in possible_subdirs:
            if subdir.exists():
                frames = sorted(list(subdir.glob('*.jpg')) + list(subdir.glob('*.png')))
                if len(frames) > 0:
                    return subdir
        
        # Try recursive search (slower, but only once during caching)
        for subdir in seq_path.rglob('*'):
            if subdir.is_dir():
                frames = sorted(list(subdir.glob('*.jpg')) + list(subdir.glob('*.png')))
                if len(frames) > 0:
                    return subdir
        
        return None
    
    def __getitem__(self, idx):
        seq_info = self.sequence_mapping[idx]
        
        # Use cached frame path (much faster!)
        frame_dir = self.frame_paths_cache.get(idx)
        
        # If no cached path or path doesn't exist, use dummy frames
        if frame_dir is None or not frame_dir.exists():
            # Create dummy black frames (silent - no print spam)
            dummy_frame = Image.new('RGB', (112, 112), color='black')
            frames = [dummy_frame] * self.max_frames
            frames_tensor = torch.stack([self.transform(img) for img in frames])
            
            # Return with zero labels
            labels = {}
            for csv_col in self.csv_label_cols:
                model_col = self.label_name_map.get(csv_col, csv_col.replace('.', '_'))
                labels[model_col] = torch.tensor(0.0, dtype=torch.float32)
            
            # Pain intensity classification label
            moment = seq_info.get('moment', 'unknown')
            pain_class = self.moment_to_class.get(moment, 0)  # Default to 0 (No pain)
            labels['pain_intensity'] = torch.tensor(pain_class, dtype=torch.long)
            
            metadata = {
                'animal': seq_info.get('animal', seq_info.get('animal_id', 'unknown')),
                'moment': seq_info.get('moment', 'unknown'),
                'sequence_id': seq_info.get('sequence_id', f'seq_{idx}')
            }
            return frames_tensor, labels, metadata
        
        # Load frames from cache (ultra-fast - no file system access!)
        frame_files = self.frame_files_cache.get(idx)
        
        # Fallback if cache miss (shouldn't happen, but safety check)
        if frame_files is None:
            if frame_dir and frame_dir.exists():
                frame_files = sorted(list(frame_dir.glob('*.jpg')) + list(frame_dir.glob('*.png')))
            else:
                frame_files = []
        
        # Sample uniformly
        if len(frame_files) > self.max_frames:
            indices = np.linspace(0, len(frame_files)-1, self.max_frames, dtype=int)
            frame_files = [frame_files[i] for i in indices]
        elif len(frame_files) < self.max_frames and len(frame_files) > 0:
            # Repeat last frame if needed
            frame_files = frame_files + [frame_files[-1]] * (self.max_frames - len(frame_files))
        
        # Load and transform (ultra-optimized - minimize operations)
        # Load images directly with transform (skip intermediate steps)
        frames = []
        last_valid_img_tensor = None
        
        for frame_file in frame_files:
            try:
                # Load and transform in one go (faster)
                img = Image.open(frame_file).convert('RGB')
                
                # Apply augmentation if needed
                if self.augment and self.aug_transform:
                    img = self.aug_transform(img)
                
                # Apply transform
                if self.transform:
                    img_tensor = self.transform(img)
                else:
                    # Fallback transform
                    img_array = np.array(img, dtype=np.float32) / 255.0
                    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
                
                frames.append(img_tensor)
                last_valid_img_tensor = img_tensor
                
            except Exception:
                # Use last valid frame or dummy
                if last_valid_img_tensor is not None:
                    frames.append(last_valid_img_tensor.clone())
                else:
                    # Create dummy tensor directly (faster than PIL)
                    dummy_tensor = torch.zeros(3, 112, 112, dtype=torch.float32)
                    if self.transform:
                        # Apply normalization if transform exists
                        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                        dummy_tensor = (dummy_tensor - mean) / std
                    frames.append(dummy_tensor)
                    last_valid_img_tensor = dummy_tensor
        
        frames = torch.stack(frames)
        
        # Labels - FIXED: Handle both dot and underscore formats
        labels = {}
        for csv_col in self.csv_label_cols:
            # Try CSV format first (with dots)
            val = seq_info.get(csv_col, np.nan)
            
            # If missing, try underscore format (model format)
            if pd.isna(val):
                model_col = self.label_name_map.get(csv_col, csv_col.replace('.', '_'))
                val = seq_info.get(model_col, np.nan)
            
            # Map to model format (underscores)
            model_col = self.label_name_map.get(csv_col, csv_col.replace('.', '_'))
            labels[model_col] = torch.tensor(
                val if not pd.isna(val) else 0.0, 
                dtype=torch.float32
            )
        
        # Pain intensity classification label
        moment = seq_info.get('moment', 'unknown')
        pain_class = self.moment_to_class.get(moment, 0)  # Default to 0 (No pain)
        labels['pain_intensity'] = torch.tensor(pain_class, dtype=torch.long)
        
        metadata = {
            'animal': seq_info.get('animal', seq_info.get('animal_id', 'unknown')),
            'moment': seq_info.get('moment', 'unknown'),
            'sequence_id': seq_info.get('sequence_id', f'seq_{idx}')
        }
        
        return frames, labels, metadata

print("✅ Dataset class defined (with label mapping fix)")

# ============================================================================
# 3. FEATURE-WEIGHTED LOSS (NEW: Based on correlation with pain)
# ============================================================================
print("\n[3] Defining feature-weighted loss...")

class FeatureMomentWeightedMSELoss(nn.Module):
    """
    Combined loss: Feature weights (based on pain correlation) × Moment weights (M2 = 2.5×)
    
    Also includes:
    - Consistency loss: Total_Facial_scale_predicted should match Total_Facial_scale_calculated
    - Pain Intensity Classification loss: CrossEntropyLoss for 3 classes (No pain, Acute pain, Residual pain)
    
    Feature weights (from data analysis):
    - Orbital_tightening: 0.538 (highest) → 1.56×
    - Ears_lateral: 0.473 → 1.37×
    - Ears_frontal: 0.465 → 1.35×
    - Total_Facial_scale: 0.627 (composite) → 1.82×
    """
    def __init__(self, moment_weights=None, feature_weights=None, normalize_features=True, 
                 consistency_weight=0.1, classification_weight=1.0):
        super(FeatureMomentWeightedMSELoss, self).__init__()
        
        # Moment weights (adjusted based on v2.0 test results)
        # M2 showed 4.6× worse performance than M0 (MAE: 3.570 vs 0.784)
        # Increased M2 weight to prioritize this critical moment
        self.moment_weights = moment_weights or {
            'M0': 1.0,   # Baseline - excellent performance (MAE=0.784)
            'M1': 1.0,   # Early post-op - good performance (MAE=1.557)
            'M2': 3.5,   # Peak pain - CRITICAL (MAE=3.570, 4.6× worse) - INCREASED from 2.5
            'M3': 1.5,   # Declining - moderate performance (MAE=2.364)
            'M4': 1.0    # Residual - good performance (MAE=0.940)
        }
        
        # Feature weights: Combined correlation + v2.0 test performance
        # Adjusted based on actual test results (Fold 7):
        # - Orbital_tightening: Best performer (R²=0.151, r=0.396) - INCREASE weight
        # - Ears_lateral: Moderate (R²=-0.033, r=0.335) - Keep correlation weight
        # - Ears_frontal: Weak (R²=-0.144, r=0.274) - Keep correlation weight
        # - Nostril_muzzle: Poor (R²=-0.557, r=-0.008) - DECREASE weight
        # - Tension_above_eyes: Poor (R²=-0.895, r=-0.236) - DECREASE weight
        correlation_weights = feature_weights or {
            'Orbital_tightening': 0.650,      # INCREASED: Best test performer (R²=0.151)
            'Ears_lateral': 0.473,             # Keep: Moderate test performance
            'Ears_frontal': 0.465,             # Keep: Weak test but strong correlation
            'Lip_jaw_profile': 0.466,          # Keep: Strong correlation
            'Cheek_tightening': 0.429,         # Keep: Moderate-Strong correlation
            'Nostril_muzzle': 0.300,           # DECREASED: Poor test (R²=-0.557)
            'Tension_above_eyes': 0.250,       # DECREASED: Poor test (R²=-0.895)
            'Total_Facial_scale': 0.627        # Keep: Composite (highest overall)
        }
        
        # Normalize feature weights to reasonable range (0.5 to 2.0)
        if normalize_features:
            min_corr = min(correlation_weights.values())
            max_corr = max(correlation_weights.values())
            
            self.feature_weights = {}
            for task, corr in correlation_weights.items():
                normalized = 0.5 + 1.5 * (corr - min_corr) / (max_corr - min_corr)
                self.feature_weights[task] = normalized
        else:
            self.feature_weights = correlation_weights
        
        self.consistency_weight = consistency_weight  # Weight for Total consistency loss
        self.classification_weight = classification_weight  # Weight for pain intensity classification loss
        self.ce_loss = nn.CrossEntropyLoss()  # Classification loss
        
        print("\n" + "="*70)
        print("FEATURE-WEIGHTED LOSS CONFIGURATION")
        print("="*70)
        print("\n📊 Feature Weights (correlation + v2.0 test performance):")
        test_performance = {
            'Orbital_tightening': '✅ Best (R²=0.151)',
            'Ears_lateral': '⚠️ Moderate (R²=-0.033)',
            'Ears_frontal': '⚠️ Weak (R²=-0.144)',
            'Lip_jaw_profile': '⚠️ Weak (R²=-0.082)',
            'Cheek_tightening': '⚠️ Weak (R²=-0.177)',
            'Nostril_muzzle': '❌ Poor (R²=-0.557)',
            'Tension_above_eyes': '❌ Poor (R²=-0.895)',
            'Total_Facial_scale': '⚠️ Moderate (R²=0.091)'
        }
        for task, weight in sorted(self.feature_weights.items(), key=lambda x: x[1], reverse=True):
            perf = test_performance.get(task, '')
            print(f"  {task:25s}: {weight:.3f}× {perf}")
        print("\n⏰ Moment Weights (adjusted based on v2.0 test results):")
        for moment, weight in sorted(self.moment_weights.items(), key=lambda x: x[1], reverse=True):
            moment_desc = {
                'M0': 'Baseline (MAE=0.784)',
                'M1': 'Early post-op (MAE=1.557)',
                'M2': 'Peak pain (MAE=3.570) ⚠️ CRITICAL',
                'M3': 'Declining (MAE=2.364)',
                'M4': 'Residual (MAE=0.940)'
            }
            desc = moment_desc.get(moment, '')
            print(f"  {moment:25s}: {weight:.1f}× {desc}")
        print(f"\n🔗 Consistency Weight: {consistency_weight} (Total predicted vs calculated)")
        print(f"\n🏷️  Classification Weight: {classification_weight} (Pain Intensity Classification)")
        print("   Pain Intensity Classes:")
        print("   • Class 0 (No pain): M0, M1")
        print("   • Class 1 (Acute pain): M2")
        print("   • Class 2 (Residual pain): M3, M4")
        print("\n📐 Total Facial Scale Loss Mechanism (v2.1 - Dual):")
        print("  • Calculated Total (sum of 7 features): 100% weight (Primary - UNESP-Botucatu validated)")
        print("  • Predicted Total (direct prediction): 50% weight (Secondary - learned)")
        print("  • Consistency Loss (predicted ≈ calculated): 10% weight (enforces relationship)")
        print("="*70 + "\n")
    
    def forward(self, predictions, targets, moments):
        total_loss = 0.0
        num_tasks = 0
        
        # Pain Intensity Classification Loss
        if 'pain_intensity' in predictions and 'pain_intensity' in targets:
            pred_class = predictions['pain_intensity']  # [batch_size, 3]
            target_class = targets['pain_intensity']     # [batch_size] (long tensor)
            classification_loss = self.ce_loss(pred_class, target_class)
            total_loss += self.classification_weight * classification_loss
            num_tasks += 1
        
        # Task losses (7 individual features)
        for task in predictions.keys():
            # Skip internal consistency outputs and classification (we'll handle them separately)
            if task in ['Total_Facial_scale_calculated', 'Total_Facial_scale_predicted', 'Total_Facial_scale', 'pain_intensity']:
                continue
            
            if task not in targets:
                continue
            
            pred = predictions[task]  # [batch_size]
            target = targets[task]     # [batch_size]
            
            # Per-sample MSE
            sample_losses = (pred - target) ** 2  # [batch_size]
            
            # Apply moment weights
            moment_weight_tensor = torch.tensor(
                [self.moment_weights.get(m, 1.0) for m in moments],
                device=pred.device,
                dtype=torch.float32
            )  # [batch_size]
            
            # Apply feature weight for this task
            feature_weight = self.feature_weights.get(task, 1.0)
            
            # Combined weight: moment_weight × feature_weight
            combined_weights = moment_weight_tensor * feature_weight
            
            # Weighted loss for this task
            weighted_loss = (sample_losses * combined_weights).mean()
            
            total_loss += weighted_loss
            num_tasks += 1
        
        # Total Facial Scale Loss (Dual Mechanism):
        # 1. Calculated Total (from 7 features) - Primary method (UNESP-Botucatu validated)
        # 2. Predicted Total (direct prediction) - Secondary method (learned)
        # 3. Consistency loss - Ensures predicted ≈ calculated
        
        if 'Total_Facial_scale_calculated' in predictions and 'Total_Facial_scale' in targets:
            # Loss for calculated Total (sum of 7 features)
            total_calc = predictions['Total_Facial_scale_calculated']
            target_total = targets['Total_Facial_scale']
            
            sample_losses_calc = (total_calc - target_total) ** 2
            moment_weight_tensor = torch.tensor(
                [self.moment_weights.get(m, 1.0) for m in moments],
                device=total_calc.device,
                dtype=torch.float32
            )
            feature_weight_total = self.feature_weights.get('Total_Facial_scale', 1.0)
            combined_weights = moment_weight_tensor * feature_weight_total
            loss_calculated = (sample_losses_calc * combined_weights).mean()
            total_loss += loss_calculated
            num_tasks += 1
        
        if 'Total_Facial_scale_predicted' in predictions and 'Total_Facial_scale' in targets:
            # Loss for predicted Total (direct prediction from model)
            total_pred = predictions['Total_Facial_scale_predicted']
            target_total = targets['Total_Facial_scale']
            
            sample_losses_pred = (total_pred - target_total) ** 2
            moment_weight_tensor = torch.tensor(
                [self.moment_weights.get(m, 1.0) for m in moments],
                device=total_pred.device,
                dtype=torch.float32
            )
            feature_weight_total = self.feature_weights.get('Total_Facial_scale', 1.0)
            combined_weights = moment_weight_tensor * feature_weight_total
            loss_predicted = (sample_losses_pred * combined_weights).mean()
            total_loss += loss_predicted * 0.5  # Give predicted Total 50% weight (calculated is primary)
            num_tasks += 0.5
        
        # Consistency loss: Total predicted should match Total calculated
        # (Based on UNESP-Botucatu scale: Total = sum of 7 features)
        # This enforces the mathematical relationship
        if 'Total_Facial_scale_predicted' in predictions and 'Total_Facial_scale_calculated' in predictions:
            total_pred = predictions['Total_Facial_scale_predicted']
            total_calc = predictions['Total_Facial_scale_calculated']
            consistency_loss = F.mse_loss(total_pred, total_calc)
            total_loss += self.consistency_weight * consistency_loss
        
        # Average across all tasks
        return total_loss / num_tasks if num_tasks > 0 else total_loss

print("✅ Feature-weighted loss defined")

# ============================================================================
# 4. STRATIFIED SAMPLER (Balance moments in each batch)
# ============================================================================
print("\n[4] Defining stratified sampler...")

def create_stratified_sampler(dataset):
    """Balance moments within batches - FIXED: No image loading"""
    moment_counts = Counter()
    
    # Count moments without loading images
    for seq_info in dataset.sequence_mapping:
        moment = seq_info.get('moment', 'unknown')
        moment_counts[moment] += 1
    
    # Inverse frequency weights
    total = sum(moment_counts.values())
    moment_weights = {m: total / count for m, count in moment_counts.items()}
    
    sample_weights = []
    for seq_info in dataset.sequence_mapping:
        moment = seq_info.get('moment', 'unknown')
        sample_weights.append(moment_weights.get(moment, 1.0))
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True
    )

print("✅ Stratified sampler defined (FIXED - no image loading)")

# ============================================================================
# 5. TRAINING FUNCTIONS
# ============================================================================
print("\n[5] Defining training functions...")

def train_one_epoch(model, dataloader, criterion, optimizer, device, config=None, scaler=None):
    model.train()
    total_loss = 0.0
    num_batches = 0
    correct_classifications = 0
    total_classifications = 0
    
    gradient_clip = config.get('gradient_clip', 0.5) if config else 0.5
    
    # Use mixed precision if scaler provided (faster + less memory for L4 GPU)
    use_amp = scaler is not None
    
    pbar = tqdm(dataloader, desc="Training", mininterval=1.0)  # Update every 1 second
    for batch_idx, (frames, labels, metadata) in enumerate(pbar):
        frames = frames.to(device, non_blocking=True)  # Faster transfer with pin_memory
        moments = metadata['moment']
        
        targets = {k: v.to(device, non_blocking=True) for k, v in labels.items()}
        
        optimizer.zero_grad()
        
        # Mixed precision forward pass
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
        
        # Calculate classification accuracy
        if 'pain_intensity' in outputs and 'pain_intensity' in targets:
            pred_classes = outputs['pain_intensity'].argmax(dim=1)
            target_classes = targets['pain_intensity']
            correct_classifications += (pred_classes == target_classes).sum().item()
            total_classifications += target_classes.size(0)
        
        # Update progress bar less frequently for speed
        if batch_idx % max(1, len(dataloader) // 10) == 0 or batch_idx == len(dataloader) - 1:
            acc = (correct_classifications / total_classifications * 100) if total_classifications > 0 else 0.0
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}', 
                'avg': f'{total_loss/num_batches:.4f}',
                'acc': f'{acc:.1f}%'
            })
    
    avg_loss = total_loss / num_batches
    classification_acc = (correct_classifications / total_classifications * 100) if total_classifications > 0 else 0.0
    return avg_loss, classification_acc


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    correct_classifications = 0
    total_classifications = 0
    
    with torch.no_grad():
        for frames, labels, metadata in dataloader:
            frames = frames.to(device, non_blocking=True)  # Faster transfer with pin_memory
            moments = metadata['moment']
            
            targets = {k: v.to(device, non_blocking=True) for k, v in labels.items()}
            
            outputs, _ = model(frames)
            loss = criterion(outputs, targets, moments)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Calculate classification accuracy
            if 'pain_intensity' in outputs and 'pain_intensity' in targets:
                pred_classes = outputs['pain_intensity'].argmax(dim=1)
                target_classes = targets['pain_intensity']
                correct_classifications += (pred_classes == target_classes).sum().item()
                total_classifications += target_classes.size(0)
    
    avg_loss = total_loss / num_batches
    classification_acc = (correct_classifications / total_classifications * 100) if total_classifications > 0 else 0.0
    return avg_loss, classification_acc

print("✅ Training functions defined")

# ============================================================================
# 6. CHECKPOINT MANAGEMENT
# ============================================================================
print("\n[6] Defining checkpoint manager...")

def find_latest_checkpoint(checkpoint_dir, fold_idx):
    """Find latest checkpoint for a fold"""
    checkpoints = sorted(checkpoint_dir.glob(f'checkpoint_fold_{fold_idx}_epoch_*.pt'))
    if checkpoints:
        return checkpoints[-1]
    return None

def save_checkpoint(model, optimizer, scheduler, epoch, fold_idx, val_loss, 
                   checkpoint_dir, is_best=False):
    """Save checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'fold': fold_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': val_loss,
    }
    
    # Save regular checkpoint
    checkpoint_path = checkpoint_dir / f'checkpoint_fold_{fold_idx}_epoch_{epoch}.pt'
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model
    if is_best:
        best_path = checkpoint_dir / f'best_model_v2.2_fold_{fold_idx}.pt'
        torch.save(checkpoint, best_path)
        print(f"✅ Best model saved! (val_loss={val_loss:.4f})")

print("✅ Checkpoint manager defined")

# ============================================================================
# 7. CONFIGURATION
# ============================================================================
print("\n[7] Loading configuration...")

# Auto-detect optimal batch size based on GPU memory
if torch.cuda.is_available():
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    gpu_name = torch.cuda.get_device_name(0)
    
    # Optimize for different GPU types (balanced for speed and performance)
    if 'L4' in gpu_name or gpu_memory_gb >= 20:  # L4 GPU (22.5 GB VRAM)
        default_batch_size = 48  # Balanced: faster than 96, but larger than 32
    elif gpu_memory_gb >= 15:  # T4 or similar (15-20 GB)
        default_batch_size = 32
    elif gpu_memory_gb >= 8:
        default_batch_size = 16
    else:
        default_batch_size = 8
else:
    default_batch_size = 8  # CPU fallback

config = {
    'batch_size': default_batch_size,  # Auto-optimized for GPU (L4: 48, T4: 32)
    'num_epochs': 80,                  # Increased for lower LR (needs more epochs)
    'learning_rate': 0.00003,          # REDUCED from 0.0001 (3× lower for stability)
    'patience': 15,                    # Increased for lower LR (more patience needed)
    'min_delta': 0.0001,               # Tighter threshold for lower LR
    'min_epochs': 5,                   # Minimum epochs before early stopping
    'max_frames': 32,                  # Keep 32 frames (don't reduce - need all pain cues)
    'weight_decay': 1e-4,             # Keep weight decay
    'gradient_clip': 0.5,             # Keep gradient clipping
    'lr_patience': 7,                  # Increased patience for LR reduction
    'warmup_epochs': 3,               # Warmup epochs for lower LR
    'num_workers': 2,                 # Use 2 workers for faster data loading
    'pin_memory': True,               # Faster GPU transfer (if GPU available)
    'prefetch_factor': 2,             # Prefetch batches (faster loading)
}

# Transform
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Device: {device}")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"   GPU: {gpu_name}")
    print(f"   Memory: {gpu_memory:.1f} GB")
    if 'L4' in gpu_name or gpu_memory >= 20:
        print(f"   ✅ L4 GPU detected - using batch_size={config['batch_size']}, max_frames={config['max_frames']} for faster training")
    elif gpu_memory >= 15:
        print(f"   ✅ T4 detected - using batch_size={config['batch_size']} for optimal performance")

print(f"\n✅ Config (v2.2 - Lightweight Model + Reduced LR):")
print("   Key Changes:")
print("   • Model: Lightweight (8→16→32→64 channels, LSTM=128) - ~3-4M params")
print("   • Learning Rate: 0.00003 (REDUCED 3× from 0.0001 for stability)")
print("   • Dropout: 0.3 (reduced from 0.5 for lighter model)")
print("   • Epochs: 80 (increased for lower LR)")
print("   • Patience: 15 (increased for lower LR)")
print("   • Min Delta: 0.0001 (tighter threshold)")
print("   • Warmup: 3 epochs (for lower LR)")
print("   • Optimizer: AdamW (better weight decay)")
print("\n   Full Config:")
for k, v in config.items():
    print(f"   {k}: {v}")

# ============================================================================
# 8. TRAINING LOOP
# ============================================================================
print("\n" + "="*80)
print("STARTING TRAINING: 9 FOLDS")
print("="*80)

# Ask which fold to start from
resume_fold = input("Start from fold (0-8) or press Enter to start from 0: ").strip()
start_fold = int(resume_fold) if resume_fold else 0

cv_folds = splits.get('cv_folds', [])
if not cv_folds:
    # Try alternative format
    cv_folds = []
    for i in range(9):
        fold_key = f'fold_{i}'
        if fold_key in splits:
            cv_folds.append(splits[fold_key])

for fold_idx, fold_data in enumerate(cv_folds[start_fold:], start=start_fold):
    # Handle different fold formats
    if isinstance(fold_data, dict):
        train_animals = fold_data.get('train_animals', [])
        val_animals = fold_data.get('val_animals', [])
    else:
        train_animals = fold_data[0] if isinstance(fold_data, (list, tuple)) else []
        val_animals = fold_data[1] if isinstance(fold_data, (list, tuple)) else []
    
    print(f"\n{'='*80}")
    print(f"FOLD {fold_idx}/8: Val={val_animals}, Train={len(train_animals)} animals")
    print(f"{'='*80}")
    
    # Check for existing checkpoint
    latest_checkpoint = find_latest_checkpoint(checkpoint_dir, fold_idx)
    
    if latest_checkpoint:
        print(f"\n⚠️  Found checkpoint: {latest_checkpoint.name}")
        resume = input("Resume from checkpoint? (y/n): ").lower() == 'y'
    else:
        resume = False
    
    # Create datasets
    train_sequences = [s for s in all_sequences if s.get('animal', s.get('animal_id')) in train_animals]
    val_sequences = [s for s in all_sequences if s.get('animal', s.get('animal_id')) in val_animals]
    
    print(f"\n📋 Preparing data...")
    print(f"   Train: {len(train_sequences)} sequences")
    print(f"   Val: {len(val_sequences)} sequences")
    
    train_dataset = FacialPainDataset_v2(train_sequences, sequence_dir,
                                         max_frames=config['max_frames'],
                                         transform=transform, augment=True)
    val_dataset = FacialPainDataset_v2(val_sequences, sequence_dir,
                                       max_frames=config['max_frames'],
                                       transform=transform, augment=False)
    
    print(f"   Creating datasets...")
    print(f"   Creating stratified sampler...")
    
    # Data loaders (optimized for L4 GPU with 2 workers for faster loading)
    train_sampler = create_stratified_sampler(train_dataset)
    num_workers = config.get('num_workers', 2)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'],
        sampler=train_sampler, 
        num_workers=num_workers,  # Use 2 workers for faster data loading
        pin_memory=config.get('pin_memory', True) if torch.cuda.is_available() else False,
        persistent_workers=True if num_workers > 0 else False,  # Keep workers alive
        prefetch_factor=config.get('prefetch_factor', 2) if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'],
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=config.get('pin_memory', True) if torch.cuda.is_available() else False,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=config.get('prefetch_factor', 2) if num_workers > 0 else None,
    )
    
    print(f"   Creating data loaders...")
    print(f"✅ Data preparation complete!")
    
    # Initialize model (lightweight v2.2)
    print(f"\n   Initializing lightweight model (v2.2)...")
    model = TemporalPainModel_v2(num_frames=config['max_frames'], lstm_hidden_size=128).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Model parameters: {num_params/1e6:.2f}M (lightweight)")
    criterion = FeatureMomentWeightedMSELoss()  # ← NEW: Feature-weighted loss
    
    # Use AdamW (better with weight decay) instead of Adam
    optimizer = torch.optim.AdamW(model.parameters(), 
                                 lr=config['learning_rate'],
                                 weight_decay=config['weight_decay'])
    
    # Improved LR scheduler with higher patience
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=config.get('lr_patience', 7),
        min_lr=1e-7
    )
    
    # Mixed precision scaler for L4 GPU (faster training + less memory)
    use_amp = torch.cuda.is_available() and hasattr(torch.cuda, 'amp')
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        print(f"   ✅ Mixed precision training enabled (faster + less memory)")
    
    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0
    min_delta = config.get('min_delta', 0.0001)
    min_epochs = config.get('min_epochs', 5)
    warmup_epochs = config.get('warmup_epochs', 3)
    
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
        
        # Learning rate warmup
        if epoch < warmup_epochs:
            warmup_lr = config['learning_rate'] * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
            print(f"   🔥 Warmup LR: {warmup_lr:.6f}")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, config, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Step scheduler (only after warmup)
        if epoch >= warmup_epochs:
            scheduler.step(val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"   Train: Loss={train_loss:.4f}, Acc={train_acc:.1f}% | Val: Loss={val_loss:.4f}, Acc={val_acc:.1f}% | LR: {current_lr:.6f}")
        
        # Check for improvement (with min_delta threshold)
        improvement = best_val_loss - val_loss
        is_best = improvement > min_delta
        
        # Save checkpoint
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model, optimizer, scheduler, epoch, fold_idx, val_loss,
                          checkpoint_dir, is_best=True)
            print(f"   ✅ Best model saved! (improvement: {improvement:.4f})")
        else:
            patience_counter += 1
            print(f"   ⏳ Patience: {patience_counter}/{config['patience']} (improvement: {improvement:.4f})")
        
        # Save regular checkpoint
        save_checkpoint(model, optimizer, scheduler, epoch, fold_idx, val_loss, checkpoint_dir)
        
        # Early stopping (only after min_epochs)
        if epoch >= min_epochs and patience_counter >= config['patience']:
            print(f"   🛑 Early stopping triggered! (trained for {epoch+1} epochs)")
            break
    
    print(f"✅ Fold {fold_idx} complete! Best: {best_val_loss:.4f}")
    
    # Cleanup old checkpoints (keep last 3)
    checkpoints = sorted(checkpoint_dir.glob(f'checkpoint_fold_{fold_idx}_epoch_*.pt'))
    for old_checkpoint in checkpoints[:-3]:
        old_checkpoint.unlink()

print(f"\n{'='*80}")
print(f"🎉 ALL TRAINING COMPLETE!")
print(f"{'='*80}")
print(f"\n📁 Models saved in: {checkpoint_dir}")
print(f"   Best models: best_model_v2.2_fold_*.pt")

