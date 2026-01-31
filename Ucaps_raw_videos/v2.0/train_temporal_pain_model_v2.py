# ============================================================================
# PHASE 2: MODEL TRAINING (v2.0 - IMPROVED)
# Optimized for Google Colab T4 GPU with Feature-Weighted Loss
# 
# Key Features:
# - Feature-weighted loss (Orbital/Ears = higher weight based on correlation)
# - Total Facial Scale calculated from 7 features (UNESP-Botucatu validated method)
# - Consistency loss to enforce Total = sum(7 features)
# - Fixed label mapping (Total.Facial.scale ↔ Total_Facial_scale)
# - Fully resumable (handles Colab disconnects)
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
print("PHASE 2: MODEL TRAINING (v2.0 - IMPROVED)")
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
    v2.0: Enhanced model with Total Facial Scale calculation
    
    Architecture:
    - 3D CNN (112×112 input, optimized for T4)
    - Bidirectional LSTM
    - Attention mechanism
    - 7 output heads (individual features)
    - Total Facial Scale = sum of 7 features (UNESP-Botucatu validated method)
    """
    def __init__(self, num_frames=32, lstm_hidden_size=256):
        super(TemporalPainModel_v2, self).__init__()
        
        # 3D CNN - optimized for 112×112 input
        self.conv1 = nn.Conv3d(3, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(16)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        
        self.conv2 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(32)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        
        self.conv3 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(64)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        
        self.conv4 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn4 = nn.BatchNorm3d(128)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        
        # CNN output: 128 × 7 × 7 = 6272
        self.cnn_output_size = 128 * 7 * 7
        
        # Bidirectional LSTM
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
        
        self.dropout = nn.Dropout(0.5)
    
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
        
        return outputs, attention_weights

print("✅ Model defined (~13.7M parameters)")

# ============================================================================
# 2. DATASET CLASS (with label mapping fix)
# ============================================================================
print("\n[2] Defining dataset class...")

class FacialPainDataset_v2(Dataset):
    """v2.0: 7 facial features + Total Facial Scale with label mapping fix"""
    def __init__(self, sequence_mapping, sequence_dir, max_frames=32, 
                 transform=None, augment=False):
        self.sequence_mapping = sequence_mapping
        self.sequence_dir = Path(sequence_dir)
        self.max_frames = max_frames
        self.transform = transform
        self.augment = augment
        
        if augment:
            self.aug_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomRotation(degrees=10),
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
    
    def __getitem__(self, idx):
        seq_info = self.sequence_mapping[idx]
        
        # Get sequence path (handle different formats)
        if 'sequence_path' in seq_info:
            seq_path = self.sequence_dir / seq_info['sequence_path']
        elif 'sequence_id' in seq_info:
            seq_path = self.sequence_dir / seq_info['sequence_id']
        else:
            raise KeyError(f"Missing sequence_path or sequence_id in sequence {idx}")
        
        # Load frames
        if not seq_path.exists():
            raise FileNotFoundError(f"Sequence path not found: {seq_path}")
        
        frame_files = sorted(list(seq_path.glob('*.jpg')) + list(seq_path.glob('*.png')))
        
        if len(frame_files) == 0:
            raise ValueError(f"No frames found in {seq_path}")
        
        # Sample uniformly
        if len(frame_files) > self.max_frames:
            indices = np.linspace(0, len(frame_files)-1, self.max_frames, dtype=int)
            frame_files = [frame_files[i] for i in indices]
        elif len(frame_files) < self.max_frames:
            frame_files = frame_files + [frame_files[-1]] * (self.max_frames - len(frame_files))
        
        # Load and transform
        frames = []
        for frame_file in frame_files:
            try:
                img = Image.open(frame_file).convert('RGB')
            except Exception as e:
                print(f"Warning: Could not load {frame_file}: {e}")
                # Use last successfully loaded frame
                if len(frames) > 0:
                    img = frames[-1]
                else:
                    raise
            
            if self.augment and self.aug_transform:
                img = self.aug_transform(img)
            
            if self.transform:
                img = self.transform(img)
            
            frames.append(img)
        
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
    
    Also includes consistency loss: Total_Facial_scale_predicted should match Total_Facial_scale_calculated
    (Based on UNESP-Botucatu scale: Total = sum of 7 features)
    
    Feature weights (from data analysis):
    - Orbital_tightening: 0.538 (highest) → 1.56×
    - Ears_lateral: 0.473 → 1.37×
    - Ears_frontal: 0.465 → 1.35×
    - Total_Facial_scale: 0.627 (composite) → 1.82×
    """
    def __init__(self, moment_weights=None, feature_weights=None, normalize_features=True, 
                 consistency_weight=0.1):
        super(FeatureMomentWeightedMSELoss, self).__init__()
        
        # Moment weights (focus on M2 acute pain)
        self.moment_weights = moment_weights or {
            'M0': 1.0,
            'M1': 1.0,
            'M2': 2.5,  # Acute pain - highest errors
            'M3': 1.5,
            'M4': 1.0
        }
        
        # Feature weights based on correlation with NRS (pain)
        correlation_weights = feature_weights or {
            'Orbital_tightening': 0.538,      # Highest correlation
            'Ears_lateral': 0.473,             # Strong
            'Ears_frontal': 0.465,             # Strong
            'Lip_jaw_profile': 0.466,          # Strong
            'Cheek_tightening': 0.429,         # Moderate-Strong
            'Nostril_muzzle': 0.374,           # Moderate
            'Tension_above_eyes': 0.345,       # Lowest (baseline)
            'Total_Facial_scale': 0.627        # Composite (highest overall)
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
        
        print("\n" + "="*70)
        print("FEATURE-WEIGHTED LOSS CONFIGURATION")
        print("="*70)
        print("\n📊 Feature Weights (based on correlation with pain):")
        for task, weight in sorted(self.feature_weights.items(), key=lambda x: x[1], reverse=True):
            print(f"  {task:25s}: {weight:.3f}×")
        print("\n⏰ Moment Weights (focus on acute pain):")
        for moment, weight in sorted(self.moment_weights.items(), key=lambda x: x[1], reverse=True):
            print(f"  {moment:25s}: {weight:.1f}×")
        print(f"\n🔗 Consistency Weight: {consistency_weight} (Total predicted vs calculated)")
        print("="*70 + "\n")
    
    def forward(self, predictions, targets, moments):
        total_loss = 0.0
        num_tasks = 0
        
        # Task losses (7 individual features + Total)
        for task in predictions.keys():
            # Skip internal consistency outputs
            if task in ['Total_Facial_scale_calculated', 'Total_Facial_scale_predicted']:
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
        
        # Consistency loss: Total predicted should match Total calculated
        # (Based on UNESP-Botucatu scale: Total = sum of 7 features)
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

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for frames, labels, metadata in pbar:
        frames = frames.to(device)
        moments = metadata['moment']
        
        targets = {k: v.to(device) for k, v in labels.items()}
        
        optimizer.zero_grad()
        outputs, _ = model(frames)
        loss = criterion(outputs, targets, moments)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / num_batches


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for frames, labels, metadata in dataloader:
            frames = frames.to(device)
            moments = metadata['moment']
            
            targets = {k: v.to(device) for k, v in labels.items()}
            
            outputs, _ = model(frames)
            loss = criterion(outputs, targets, moments)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches

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
        best_path = checkpoint_dir / f'best_model_v2_fold_{fold_idx}.pt'
        torch.save(checkpoint, best_path)
        print(f"✅ Best model saved! (val_loss={val_loss:.4f})")

print("✅ Checkpoint manager defined")

# ============================================================================
# 7. CONFIGURATION
# ============================================================================
print("\n[7] Loading configuration...")

config = {
    'batch_size': 32,           # T4-optimized (16GB VRAM)
    'num_epochs': 50,
    'learning_rate': 0.0001,
    'patience': 10,
    'max_frames': 32,
    'weight_decay': 1e-5,
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
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

print(f"\n✅ Config (T4-optimized):")
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
    
    # Data loaders
    train_sampler = create_stratified_sampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                             sampler=train_sampler, num_workers=0)  # num_workers=0 for Colab
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'],
                           shuffle=False, num_workers=0)
    
    print(f"   Creating data loaders...")
    print(f"✅ Data preparation complete!")
    
    # Initialize model
    print(f"\n   Initializing model...")
    model = TemporalPainModel_v2(num_frames=config['max_frames']).to(device)
    criterion = FeatureMomentWeightedMSELoss()  # ← NEW: Feature-weighted loss
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=config['learning_rate'],
                                 weight_decay=config['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)  # Removed verbose (not in newer PyTorch)
    
    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0
    
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
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        print(f"   Train: {train_loss:.4f}, Val: {val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model, optimizer, scheduler, epoch, fold_idx, val_loss,
                          checkpoint_dir, is_best=True)
        else:
            patience_counter += 1
            print(f"⏳ Patience: {patience_counter}/{config['patience']}")
        
        # Save regular checkpoint
        save_checkpoint(model, optimizer, scheduler, epoch, fold_idx, val_loss, checkpoint_dir)
        
        # Early stopping
        if patience_counter >= config['patience']:
            print(f"🛑 Early stopping triggered!")
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
print(f"   Best models: best_model_v2_fold_*.pt")

