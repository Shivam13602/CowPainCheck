# ============================================================================
# VALIDATION RESULTS ANALYZER - v2.5 (Dual Classification)
# Analyzes validation results across all folds for dual classification tasks
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import json
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import warnings
import os
from tqdm import tqdm
warnings.filterwarnings('ignore')

# ============================================================================
# SETUP
# ============================================================================
print("="*80)
print("VALIDATION RESULTS ANALYZER - v2.5 (Dual Classification)")
print("="*80)

# Mount Drive (for Colab)
try:
    from google.colab import drive
    drive.mount('/content/drive')
    base_path = Path('/content/drive/MyDrive')
except ImportError:
    base_path = Path(os.getcwd()).parent
    print(f"Running locally - using base_path: {base_path}")

project_dir = base_path / 'facial_pain_project_v2'
sequence_dir = base_path / 'sequence'
checkpoint_dir = project_dir / 'checkpoints_v2.5'

splits_file = project_dir / 'train_val_test_splits_v2.json'
mapping_file = project_dir / 'sequence_label_mapping_v2.json'

print(f"\n📁 Paths:")
print(f"   Checkpoint dir: {checkpoint_dir} {'✅' if checkpoint_dir.exists() else '❌'}")
print(f"   Splits file: {'✅' if splits_file.exists() else '❌'}")
print(f"   Mapping file: {'✅' if mapping_file.exists() else '❌'}")

# Load splits and mappings
with open(splits_file, 'r') as f:
    splits = json.load(f)
with open(mapping_file, 'r') as f:
    sequence_mapping = json.load(f)

if isinstance(sequence_mapping, dict):
    if 'sequences' in sequence_mapping:
        all_sequences = sequence_mapping['sequences']
    else:
        all_sequences = [{'sequence_id': k, **v} for k, v in sequence_mapping.items()]
else:
    all_sequences = sequence_mapping

print(f"✅ Loaded {len(all_sequences)} sequences")

# ============================================================================
# MODEL ARCHITECTURE (v2.5)
# ============================================================================
class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output):
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        return context_vector, attention_weights


class TemporalPainModel_v2_5(nn.Module):
    def __init__(self, num_frames=32, lstm_hidden_size=128, use_bidirectional=False):
        super(TemporalPainModel_v2_5, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.cnn_output_size = 256
        self.use_bidirectional = use_bidirectional
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_size,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=use_bidirectional
        )
        
        self.lstm_output_size = lstm_hidden_size * 2 if use_bidirectional else lstm_hidden_size
        self.attention = AttentionLayer(self.lstm_output_size)
        
        self.pain_classification_head = nn.Sequential(
            nn.Linear(self.lstm_output_size, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
        
        self.intensity_classification_head = nn.Sequential(
            nn.Linear(self.lstm_output_size, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 3)
        )
        
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        batch_size, num_frames, C, H, W = x.size()
        x = x.view(batch_size * num_frames, C, H, W)
        cnn_features = self.cnn(x)
        cnn_features = cnn_features.view(batch_size * num_frames, -1)
        cnn_features = cnn_features.view(batch_size, num_frames, self.cnn_output_size)
        lstm_out, _ = self.lstm(cnn_features)
        context, attention_weights = self.attention(lstm_out)
        context = self.dropout(context)
        
        outputs = {}
        outputs['pain_classification'] = self.pain_classification_head(context).squeeze(-1)
        outputs['intensity_classification'] = self.intensity_classification_head(context)
        
        return outputs, attention_weights

# ============================================================================
# DATASET CLASS (v2.5)
# ============================================================================
class FacialPainDataset_v2_5(Dataset):
    def __init__(self, sequence_mapping, sequence_dir, max_frames=32, transform=None, augment=False,
                 global_cache=None):
        self.sequence_mapping = sequence_mapping
        self.sequence_dir = Path(sequence_dir)
        self.max_frames = max_frames
        self.transform = transform
        self.augment = augment
        self.global_cache = global_cache  # Cache keyed by sequence_id

        # If no global cache provided, create empty cache (will be populated on-demand)
        if self.global_cache is None:
            self.global_cache = {}

    def _get_cache_key(self, seq_info):
        """Generate a unique cache key for a sequence"""
        seq_id = seq_info.get('sequence_id')
        animal = seq_info.get('animal', seq_info.get('animal_id', 'unknown'))
        moment = seq_info.get('moment', 'unknown')
        return f"{seq_id}_{animal}_{moment}"

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
        cache_key = self._get_cache_key(seq_info)
        
        # Try to get from global cache
        if cache_key in self.global_cache:
            frame_dir = self.global_cache[cache_key].get('path')
            frame_files = self.global_cache[cache_key].get('files')
        else:
            # Cache miss - find path and cache it
            frame_dir = self._find_frames_path(seq_info)
            if frame_dir and frame_dir.exists():
                frame_files = sorted(list(frame_dir.glob('*.jpg')) + list(frame_dir.glob('*.png')))
                frame_files = frame_files if len(frame_files) > 0 else None
            else:
                frame_files = None
            
            # Store in global cache
            self.global_cache[cache_key] = {'path': frame_dir, 'files': frame_files}

        if frame_dir is None or (frame_dir is not None and not frame_dir.exists()):
            dummy_frame = Image.new('RGB', (112, 112), color='black')
            frames = [dummy_frame] * self.max_frames
            frames_tensor = torch.stack([self.transform(img) for img in frames])
            labels = {}
            labels['pain_classification'] = torch.tensor(0.0, dtype=torch.float32)
            labels['intensity_classification'] = torch.tensor(0, dtype=torch.long)
            metadata = {
                'animal': seq_info.get('animal', 'unknown'),
                'moment': seq_info.get('moment', 'unknown'),
                'sequence_id': seq_info.get('sequence_id', f'seq_{idx}')
            }
            return frames_tensor, labels, metadata

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

        moment = metadata['moment']
        is_pain = 1.0 if moment in ['M2', 'M3', 'M4'] else 0.0
        
        if moment in ['M0', 'M1']:
            intensity_class = 0
        elif moment == 'M2':
            intensity_class = 1
        elif moment in ['M3', 'M4']:
            intensity_class = 2
        else:
            intensity_class = 0
        
        labels = {}
        labels['pain_classification'] = torch.tensor(is_pain, dtype=torch.float32)
        labels['intensity_classification'] = torch.tensor(intensity_class, dtype=torch.long)

        return frames, labels, metadata

# ============================================================================
# PATH CACHING FUNCTION (Global Cache)
# ============================================================================
def create_global_path_cache(sequence_mapping, sequence_dir):
    """Create a global path cache for all sequences (one-time operation)
    Cache is keyed by sequence_id for easy lookups across different dataset subsets
    """
    print("\n📁 Creating global path cache for all sequences (one-time)...")
    global_cache = {}
    sequence_dir = Path(sequence_dir)
    
    for seq_info in tqdm(sequence_mapping, desc="Caching paths"):
        # Generate unique cache key
        seq_id = seq_info.get('sequence_id')
        animal = seq_info.get('animal', seq_info.get('animal_id', 'unknown'))
        moment = seq_info.get('moment', 'unknown')
        cache_key = f"{seq_id}_{animal}_{moment}"
        
        # Skip if already cached
        if cache_key in global_cache:
            continue
            
        # Find frame path
        if 'sequence_path' in seq_info:
            seq_path = sequence_dir / seq_info['sequence_path']
        elif 'sequence_id' in seq_info:
            seq_path = sequence_dir / seq_info['sequence_id']
        else:
            global_cache[cache_key] = {'path': None, 'files': None}
            continue
            
        frame_path = None
        if seq_path.exists():
            frame_files = sorted(list(seq_path.glob('*.jpg')) + list(seq_path.glob('*.png')))
            if len(frame_files) > 0:
                frame_path = seq_path
        else:
            possible_subdirs = [seq_path / d for d in ['sequence_001', 'frames', 'images']]
            for subdir in possible_subdirs:
                if subdir.exists():
                    frames = sorted(list(subdir.glob('*.jpg')) + list(subdir.glob('*.png')))
                    if len(frames) > 0:
                        frame_path = subdir
                        break
            if frame_path is None:
                for subdir in seq_path.rglob('*'):
                    if subdir.is_dir():
                        frames = sorted(list(subdir.glob('*.jpg')) + list(subdir.glob('*.png')))
                        if len(frames) > 0:
                            frame_path = subdir
                            break
        
        # Cache the path and files
        if frame_path and frame_path.exists():
            frame_files = sorted(list(frame_path.glob('*.jpg')) + list(frame_path.glob('*.png')))
            frame_files = frame_files if len(frame_files) > 0 else None
        else:
            frame_files = None
        
        global_cache[cache_key] = {'path': frame_path, 'files': frame_files}
    
    print(f"   ✅ Cached {len(global_cache)} sequence paths")
    return global_cache

# ============================================================================
# EVALUATION FUNCTION
# ============================================================================
def evaluate_fold(fold_idx, checkpoint_dir, sequence_dir, all_sequences, splits, device,
                  global_cache=None):
    """Evaluate a single fold on its validation set"""
    
    cv_folds = splits.get('cv_folds', [])
    if not cv_folds:
        cv_folds = [splits[f'fold_{i}'] for i in range(9) if f'fold_{i}' in splits]
    
    if fold_idx >= len(cv_folds):
        return None
    
    fold_data = cv_folds[fold_idx]
    if isinstance(fold_data, dict):
        val_animals = fold_data.get('val_animals', [])
    else:
        val_animals = fold_data[1]
    
    val_sequences = [s for s in all_sequences if s.get('animal', s.get('animal_id')) in val_animals]
    
    if len(val_sequences) == 0:
        return None
    
    best_model_path = checkpoint_dir / f'best_model_v2.5_fold_{fold_idx}.pt'
    if not best_model_path.exists():
        print(f"   ⚠️  Best model not found for fold {fold_idx}")
        return None
    
    checkpoint = torch.load(best_model_path, map_location=device)
    val_loss = checkpoint.get('val_loss', None)
    epoch = checkpoint.get('epoch', None)
    
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Use global cache (shared across all folds)
    val_dataset = FacialPainDataset_v2_5(
        val_sequences, sequence_dir, max_frames=32, transform=transform,
        global_cache=global_cache
    )
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    model = TemporalPainModel_v2_5(num_frames=32, lstm_hidden_size=128, use_bidirectional=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Collect predictions for both tasks
    all_pain_preds = []
    all_pain_targets = []
    all_intensity_preds = []
    all_intensity_targets = []
    all_moments = []
    
    with torch.no_grad():
        for frames, labels, metadata in val_loader:
            frames = frames.to(device)
            moments = metadata['moment']
            
            outputs, _ = model(frames)
            
            # Task 1: Binary classification
            pred_pain_logits = outputs['pain_classification']
            target_pain = labels['pain_classification'].to(device)
            pred_pain_binary = (torch.sigmoid(pred_pain_logits) > 0.5).int()
            all_pain_preds.append(pred_pain_binary.cpu())
            all_pain_targets.append(target_pain.cpu())
            
            # Task 2: 3-Class classification
            pred_intensity_logits = outputs['intensity_classification']
            target_intensity = labels['intensity_classification'].to(device)
            pred_intensity_class = torch.argmax(pred_intensity_logits, dim=1)
            all_intensity_preds.append(pred_intensity_class.cpu())
            all_intensity_targets.append(target_intensity.cpu())
            
            all_moments.extend(moments)
    
    # Convert to numpy
    all_pain_preds = torch.cat(all_pain_preds).numpy()
    all_pain_targets = torch.cat(all_pain_targets).numpy()
    all_intensity_preds = torch.cat(all_intensity_preds).numpy()
    all_intensity_targets = torch.cat(all_intensity_targets).numpy()
    
    # Compute metrics for Task 1 (Binary)
    task1_acc = accuracy_score(all_pain_targets, all_pain_preds)
    task1_f1 = f1_score(all_pain_targets, all_pain_preds, zero_division=0)
    task1_precision = precision_score(all_pain_targets, all_pain_preds, zero_division=0)
    task1_recall = recall_score(all_pain_targets, all_pain_preds, zero_division=0)
    
    # Compute metrics for Task 2 (3-Class)
    task2_acc = accuracy_score(all_intensity_targets, all_intensity_preds)
    task2_f1 = f1_score(all_intensity_targets, all_intensity_preds, average='weighted', zero_division=0)
    task2_precision = precision_score(all_intensity_targets, all_intensity_preds, average='weighted', zero_division=0)
    task2_recall = recall_score(all_intensity_targets, all_intensity_preds, average='weighted', zero_division=0)
    
    task1_metrics = {
        'Accuracy': task1_acc,
        'F1': task1_f1,
        'Precision': task1_precision,
        'Recall': task1_recall,
        'N': len(all_pain_targets),
        'val_loss': val_loss,
        'epoch': epoch
    }
    
    task2_metrics = {
        'Accuracy': task2_acc,
        'F1': task2_f1,
        'Precision': task2_precision,
        'Recall': task2_recall,
        'N': len(all_intensity_targets),
        'val_loss': val_loss,
        'epoch': epoch
    }
    
    # Moment-wise metrics
    moment_metrics = {}
    for moment in ['M0', 'M1', 'M2', 'M3', 'M4']:
        mask = np.array(all_moments) == moment
        if mask.sum() == 0:
            continue
        
        # Task 1 per moment
        pain_pred_moment = all_pain_preds[mask]
        pain_target_moment = all_pain_targets[mask]
        
        # Task 2 per moment
        intensity_pred_moment = all_intensity_preds[mask]
        intensity_target_moment = all_intensity_targets[mask]
        
        if len(pain_pred_moment) > 0:
            moment_acc_task1 = accuracy_score(pain_target_moment, pain_pred_moment)
            moment_f1_task1 = f1_score(pain_target_moment, pain_pred_moment, zero_division=0)
        else:
            moment_acc_task1 = 0.0
            moment_f1_task1 = 0.0
        
        if len(intensity_pred_moment) > 0:
            moment_acc_task2 = accuracy_score(intensity_target_moment, intensity_pred_moment)
            moment_f1_task2 = f1_score(intensity_target_moment, intensity_pred_moment, average='weighted', zero_division=0)
        else:
            moment_acc_task2 = 0.0
            moment_f1_task2 = 0.0
        
        moment_metrics[moment] = {
            'Task1_Accuracy': moment_acc_task1,
            'Task1_F1': moment_f1_task1,
            'Task2_Accuracy': moment_acc_task2,
            'Task2_F1': moment_f1_task2,
            'N': len(pain_pred_moment)
        }
    
    return {
        'fold': fold_idx,
        'val_animals': val_animals,
        'n_sequences': len(val_sequences),
        'task1_metrics': task1_metrics,
        'task2_metrics': task2_metrics,
        'moment_metrics': moment_metrics
    }

# ============================================================================
# MAIN ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("ANALYZING VALIDATION RESULTS ACROSS ALL FOLDS")
print("="*80)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n✅ Device: {device}")

# Optional: Training results from training output (if available)
# Can be updated with actual training metrics
training_results = {
    2: {'val_loss': 1.8037, 'task1_f1': 0.917, 'task2_f1': 0.786},
    3: {'val_loss': 2.8864, 'task1_f1': 0.773, 'task2_f1': 0.482},
    4: {'val_loss': 2.5173, 'task1_f1': 0.850, 'task2_f1': 0.543},
    5: {'val_loss': 2.3282, 'task1_f1': 0.837, 'task2_f1': 0.527},
    6: {'val_loss': 2.3083, 'task1_f1': 0.868, 'task2_f1': 0.690},
    7: {'val_loss': 3.0926, 'task1_f1': 0.831, 'task2_f1': 0.470},
    8: {'val_loss': 2.9214, 'task1_f1': 0.615, 'task2_f1': 0.505}
}

# Check which folds have checkpoints available
available_folds = []
for fold_idx in range(9):
    best_model_path = checkpoint_dir / f'best_model_v2.5_fold_{fold_idx}.pt'
    if best_model_path.exists():
        available_folds.append(fold_idx)

print(f"\n📊 Found checkpoints for folds: {sorted(available_folds)}")

# Display training results summary if available
if training_results:
    print("\n📊 VALIDATION LOSS SUMMARY (from training output - if available):")
    print("-" * 80)
    print(f"{'Fold':<6} {'Val Loss':<12} {'Task 1 F1':<12} {'Task 2 F1':<12} {'Status'}")
    print("-" * 80)
    for fold_idx in sorted(training_results.keys()):
        result = training_results[fold_idx]
        val_loss = result['val_loss']
        task1_f1 = result['task1_f1']
        task2_f1 = result['task2_f1']
        
        if val_loss < 2.0:
            status = "✅ Excellent"
        elif val_loss < 2.5:
            status = "✅ Good"
        elif val_loss < 3.0:
            status = "⚠️  Moderate"
        else:
            status = "❌ Poor"
        
        print(f"{fold_idx:<6} {val_loss:<12.4f} {task1_f1:<12.3f} {task2_f1:<12.3f} {status}")
    
    if training_results:
        mean_val_loss = np.mean([r['val_loss'] for r in training_results.values()])
        std_val_loss = np.std([r['val_loss'] for r in training_results.values()])
        mean_task1_f1 = np.mean([r['task1_f1'] for r in training_results.values()])
        mean_task2_f1 = np.mean([r['task2_f1'] for r in training_results.values()])
        
        print("-" * 80)
        print(f"{'Mean':<6} {mean_val_loss:<12.4f} {'':<12} {'':<12}")
        print(f"{'Std':<6} {std_val_loss:<12.4f}")
        print(f"{'Mean Task 1 F1':<18} {mean_task1_f1:<12.3f}")
        print(f"{'Mean Task 2 F1':<18} {mean_task2_f1:<12.3f}")

# Create global path cache once for all sequences
print("\n" + "="*80)
print("PREPARING GLOBAL PATH CACHE")
print("="*80)
global_cache = create_global_path_cache(all_sequences, sequence_dir)

# Evaluate each fold
print("\n" + "="*80)
print("EVALUATING MODELS ON VALIDATION SETS")
print("="*80)

all_fold_results = []
for fold_idx in range(9):  # Check all folds 0-8
    # Check if checkpoint exists
    best_model_path = checkpoint_dir / f'best_model_v2.5_fold_{fold_idx}.pt'
    if not best_model_path.exists():
        print(f"\n📊 Fold {fold_idx}: ⚠️  Checkpoint not found, skipping...")
        continue
    
    print(f"\n📊 Evaluating Fold {fold_idx}...")
    try:
        result = evaluate_fold(fold_idx, checkpoint_dir, sequence_dir, all_sequences, splits, device,
                              global_cache)
        if result:
            all_fold_results.append(result)
            t1 = result['task1_metrics']
            t2 = result['task2_metrics']
            print(f"   ✅ Fold {fold_idx}: {result['n_sequences']} sequences")
            print(f"      Task 1: Acc={t1['Accuracy']:.3f}, F1={t1['F1']:.3f}")
            print(f"      Task 2: Acc={t2['Accuracy']:.3f}, F1={t2['F1']:.3f}")
        else:
            print(f"   ⚠️  Fold {fold_idx}: Could not evaluate")
    except Exception as e:
        print(f"   ❌ Fold {fold_idx}: Error - {e}")
        import traceback
        traceback.print_exc()

if len(all_fold_results) == 0:
    print("\n⚠️  No fold results available. Check checkpoint paths.")
else:
    # Aggregate results
    print("\n" + "="*80)
    print("AGGREGATE VALIDATION RESULTS - Task 1 (Binary Classification)")
    print("="*80)
    
    task1_summary = []
    for result in all_fold_results:
        m = result['task1_metrics']
        task1_summary.append({
            'Fold': result['fold'],
            'N': m['N'],
            'Accuracy': m['Accuracy'],
            'F1': m['F1'],
            'Precision': m['Precision'],
            'Recall': m['Recall'],
            'Val_Loss': m['val_loss']
        })
    
    if task1_summary:
        df1 = pd.DataFrame(task1_summary)
        print("\n📊 Per-Fold Performance - Task 1 (Binary):")
        print(df1.to_string(index=False))
        
        print("\n📊 Aggregate Statistics - Task 1:")
        print("-" * 80)
        print(f"{'Metric':<15} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
        print("-" * 80)
        for metric in ['Accuracy', 'F1', 'Precision', 'Recall']:
            values = df1[metric]
            print(f"{metric:<15} {values.mean():<12.3f} {values.std():<12.3f} {values.min():<12.3f} {values.max():<12.3f}")
    
    print("\n" + "="*80)
    print("AGGREGATE VALIDATION RESULTS - Task 2 (3-Class Classification)")
    print("="*80)
    
    task2_summary = []
    for result in all_fold_results:
        m = result['task2_metrics']
        task2_summary.append({
            'Fold': result['fold'],
            'N': m['N'],
            'Accuracy': m['Accuracy'],
            'F1': m['F1'],
            'Precision': m['Precision'],
            'Recall': m['Recall'],
            'Val_Loss': m['val_loss']
        })
    
    if task2_summary:
        df2 = pd.DataFrame(task2_summary)
        print("\n📊 Per-Fold Performance - Task 2 (3-Class):")
        print(df2.to_string(index=False))
        
        print("\n📊 Aggregate Statistics - Task 2:")
        print("-" * 80)
        print(f"{'Metric':<15} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
        print("-" * 80)
        for metric in ['Accuracy', 'F1', 'Precision', 'Recall']:
            values = df2[metric]
            print(f"{metric:<15} {values.mean():<12.3f} {values.std():<12.3f} {values.min():<12.3f} {values.max():<12.3f}")
    
    # Moment-wise aggregate
    print("\n" + "="*80)
    print("MOMENT-WISE AGGREGATE RESULTS")
    print("="*80)
    
    moment_aggregates = defaultdict(lambda: {'Task1_Accuracy': [], 'Task1_F1': [], 'Task2_Accuracy': [], 'Task2_F1': [], 'N': []})
    for result in all_fold_results:
        if 'moment_metrics' in result:
            for moment, metrics in result['moment_metrics'].items():
                moment_aggregates[moment]['Task1_Accuracy'].append(metrics['Task1_Accuracy'])
                moment_aggregates[moment]['Task1_F1'].append(metrics['Task1_F1'])
                moment_aggregates[moment]['Task2_Accuracy'].append(metrics['Task2_Accuracy'])
                moment_aggregates[moment]['Task2_F1'].append(metrics['Task2_F1'])
                moment_aggregates[moment]['N'].append(metrics['N'])
    
    print(f"\n{'Moment':<10} {'Task1_Acc':<12} {'Task1_F1':<12} {'Task2_Acc':<12} {'Task2_F1':<12} {'Total N':<10}")
    print("-" * 80)
    for moment in ['M0', 'M1', 'M2', 'M3', 'M4']:
        if moment in moment_aggregates:
            agg = moment_aggregates[moment]
            mean_acc1 = np.mean(agg['Task1_Accuracy'])
            mean_f1_1 = np.mean(agg['Task1_F1'])
            mean_acc2 = np.mean(agg['Task2_Accuracy'])
            mean_f1_2 = np.mean(agg['Task2_F1'])
            total_n = sum(agg['N'])
            print(f"{moment:<10} {mean_acc1:<12.3f} {mean_f1_1:<12.3f} {mean_acc2:<12.3f} {mean_f1_2:<12.3f} {total_n:<10}")

print("\n" + "="*80)
print("✅ VALIDATION ANALYSIS COMPLETE")
print("="*80)
print("\n💡 Next Step: Run test set evaluation script")
print("   Use: evaluate_test_set_v2.5.py")

