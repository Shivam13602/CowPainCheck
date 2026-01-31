# ============================================================================
# TEST RESULTS VISUALIZATION - v2.4 (Regression Only)
# Creates comprehensive visualizations of test results in Colab
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
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
import os
from typing import Dict, List, Tuple, Optional
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# SETUP
# ============================================================================
print("="*80)
print("TEST RESULTS VISUALIZATION - v2.4 (Regression Only)")
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
checkpoint_dir = project_dir / 'checkpoints_v2.4'

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

# Get test animals
test_animals = splits.get('test_animals', [14, 17])
test_sequences = [s for s in all_sequences if s.get('animal', s.get('animal_id')) in test_animals]

# ============================================================================
# MODEL ARCHITECTURE (Same as training)
# ============================================================================
class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output):
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        return context_vector, attention_weights

class TemporalPainModel_v2_4(nn.Module):
    def __init__(self, num_frames=32, lstm_hidden_size=128, use_bidirectional=False):
        super(TemporalPainModel_v2_4, self).__init__()
        
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
        
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_size,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=use_bidirectional
        )
        self.lstm_output_size = lstm_hidden_size * (2 if use_bidirectional else 1)
        
        self.attention = AttentionLayer(self.lstm_output_size)
        
        self.output_heads = nn.ModuleDict({
            'Orbital_tightening': nn.Linear(self.lstm_output_size, 1),
            'Tension_above_eyes': nn.Linear(self.lstm_output_size, 1),
            'Cheek_tightening': nn.Linear(self.lstm_output_size, 1),
            'Ears_frontal': nn.Linear(self.lstm_output_size, 1),
            'Ears_lateral': nn.Linear(self.lstm_output_size, 1),
            'Lip_jaw_profile': nn.Linear(self.lstm_output_size, 1),
            'Nostril_muzzle': nn.Linear(self.lstm_output_size, 1),
        })
        
        self.total_head = nn.Linear(self.lstm_output_size, 1)
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
        for task, head in self.output_heads.items():
            outputs[task] = head(context).squeeze(-1)
        
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
        outputs['Total_Facial_scale_predicted'] = self.total_head(context).squeeze(-1)
        outputs['Total_Facial_scale'] = outputs['Total_Facial_scale_calculated']
        
        return outputs, attention_weights

# ============================================================================
# DATASET CLASS
# ============================================================================
class FacialPainDataset_v2_4(Dataset):
    def __init__(self, sequence_mapping, sequence_dir, max_frames=32, transform=None):
        self.sequence_mapping = sequence_mapping
        self.sequence_dir = Path(sequence_dir)
        self.max_frames = max_frames
        self.transform = transform

        self.frame_paths_cache = {}
        self.frame_files_cache = {}
        for idx, seq_info in enumerate(sequence_mapping):
            frame_path = self._find_frames_path(seq_info)
            self.frame_paths_cache[idx] = frame_path
            if frame_path and frame_path.exists():
                frame_files = sorted(list(frame_path.glob('*.jpg')) + list(frame_path.glob('*.png')))
                self.frame_files_cache[idx] = frame_files if len(frame_files) > 0 else None
            else:
                self.frame_files_cache[idx] = None

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

# ============================================================================
# EVALUATION FUNCTION
# ============================================================================
def evaluate_model(model, dataloader, device):
    """Evaluate model on test set"""
    model.eval()
    
    all_predictions = defaultdict(list)
    all_targets = defaultdict(list)
    all_moments = []
    all_animals = []
    
    with torch.no_grad():
        for frames, labels, metadata in dataloader:
            frames = frames.to(device)
            moments = metadata['moment']
            animals = metadata['animal']
            
            outputs, _ = model(frames)
            
            for task in outputs.keys():
                if task in ['Total_Facial_scale_calculated', 'Total_Facial_scale_predicted']:
                    continue
                pred = outputs[task].cpu().numpy()
                target = labels[task].cpu().numpy()
                all_predictions[task].extend(pred)
                all_targets[task].extend(target)
            
            all_moments.extend(moments)
            all_animals.extend(animals)
    
    # Convert to numpy
    for task in all_predictions.keys():
        all_predictions[task] = np.array(all_predictions[task])
        all_targets[task] = np.array(all_targets[task])
    
    return all_predictions, all_targets, all_moments, all_animals

def compute_metrics(predictions, targets):
    """Compute comprehensive metrics"""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from scipy.stats import pearsonr
    
    metrics = {}
    for task in predictions.keys():
        pred = predictions[task]
        target = targets[task]
        
        mae = mean_absolute_error(target, pred)
        rmse = np.sqrt(mean_squared_error(target, pred))
        r2 = r2_score(target, pred)
        
        try:
            r, p = pearsonr(target, pred)
        except:
            r, p = np.nan, np.nan
        
        metrics[task] = {
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2,
            'r': r,
            'p_value': p,
            'N': len(pred)
        }
    
    return metrics

# ============================================================================
# EVALUATE ALL FOLDS
# ============================================================================
print("\n" + "="*80)
print("EVALUATING ALL FOLDS ON TEST SET")
print("="*80)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n✅ Device: {device}")

# Create test dataset
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_dataset = FacialPainDataset_v2_4(test_sequences, sequence_dir, max_frames=32, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

# Evaluate with each fold model
all_fold_results = []
for fold_idx in range(9):
    print(f"\n📊 Evaluating Fold {fold_idx}...")
    
    best_model_path = checkpoint_dir / f'best_model_v2.4_fold_{fold_idx}.pt'
    if not best_model_path.exists():
        print(f"   ⚠️  Model not found: {best_model_path.name}")
        continue
    
    # Load model
    checkpoint = torch.load(best_model_path, map_location=device)
    model = TemporalPainModel_v2_4(num_frames=32, lstm_hidden_size=128, use_bidirectional=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Evaluate
    predictions, targets, moments, animals = evaluate_model(model, test_loader, device)
    metrics = compute_metrics(predictions, targets)
    
    all_fold_results.append({
        'fold': fold_idx,
        'predictions': predictions,
        'targets': targets,
        'moments': moments,
        'animals': animals,
        'metrics': metrics
    })

# Ensemble predictions
ensemble_predictions = defaultdict(list)
ensemble_targets = defaultdict(list)
ensemble_moments = []
ensemble_animals = []

for result in all_fold_results:
    for task in result['predictions'].keys():
        ensemble_predictions[task].append(result['predictions'][task])
    ensemble_targets = result['targets']
    ensemble_moments = result['moments']
    ensemble_animals = result['animals']

for task in ensemble_predictions.keys():
    ensemble_predictions[task] = np.mean(ensemble_predictions[task], axis=0)

ensemble_metrics = compute_metrics(ensemble_predictions, ensemble_targets)

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_per_fold_results(all_fold_results, ensemble_metrics):
    """Plot per-fold performance"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Per-Fold Test Performance - Total_Facial_scale', fontsize=16, fontweight='bold')
    
    folds = [r['fold'] for r in all_fold_results]
    mae_values = [r['metrics']['Total_Facial_scale']['MAE'] for r in all_fold_results]
    rmse_values = [r['metrics']['Total_Facial_scale']['RMSE'] for r in all_fold_results]
    r2_values = [r['metrics']['Total_Facial_scale']['R²'] for r in all_fold_results]
    r_values = [r['metrics']['Total_Facial_scale']['r'] for r in all_fold_results]
    
    # MAE
    axes[0, 0].bar(folds, mae_values, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0, 0].axhline(y=ensemble_metrics['Total_Facial_scale']['MAE'], 
                      color='red', linestyle='--', linewidth=2, label='Ensemble')
    axes[0, 0].set_xlabel('Fold', fontsize=12)
    axes[0, 0].set_ylabel('MAE', fontsize=12)
    axes[0, 0].set_title('Mean Absolute Error', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # RMSE
    axes[0, 1].bar(folds, rmse_values, color='coral', alpha=0.7, edgecolor='black')
    axes[0, 1].axhline(y=ensemble_metrics['Total_Facial_scale']['RMSE'], 
                      color='red', linestyle='--', linewidth=2, label='Ensemble')
    axes[0, 1].set_xlabel('Fold', fontsize=12)
    axes[0, 1].set_ylabel('RMSE', fontsize=12)
    axes[0, 1].set_title('Root Mean Squared Error', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # R²
    axes[1, 0].bar(folds, r2_values, color='mediumseagreen', alpha=0.7, edgecolor='black')
    axes[1, 0].axhline(y=ensemble_metrics['Total_Facial_scale']['R²'], 
                      color='red', linestyle='--', linewidth=2, label='Ensemble')
    axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1, 0].set_xlabel('Fold', fontsize=12)
    axes[1, 0].set_ylabel('R²', fontsize=12)
    axes[1, 0].set_title('Coefficient of Determination (R²)', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Correlation
    axes[1, 1].bar(folds, r_values, color='gold', alpha=0.7, edgecolor='black')
    axes[1, 1].axhline(y=ensemble_metrics['Total_Facial_scale']['r'], 
                      color='red', linestyle='--', linewidth=2, label='Ensemble')
    axes[1, 1].set_xlabel('Fold', fontsize=12)
    axes[1, 1].set_ylabel('Pearson r', fontsize=12)
    axes[1, 1].set_title('Pearson Correlation', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_per_moment_results(ensemble_predictions, ensemble_targets, ensemble_moments):
    """Plot per-moment performance"""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from scipy.stats import pearsonr
    
    moments = ['M0', 'M1', 'M2', 'M3', 'M4']
    moment_metrics = {}
    
    for moment in moments:
        mask = np.array(ensemble_moments) == moment
        if mask.sum() == 0:
            continue
        
        pred = ensemble_predictions['Total_Facial_scale'][mask]
        target = ensemble_targets['Total_Facial_scale'][mask]
        
        mae = mean_absolute_error(target, pred)
        rmse = np.sqrt(mean_squared_error(target, pred))
        r2 = r2_score(target, pred)
        try:
            r, _ = pearsonr(target, pred)
        except:
            r = np.nan
        
        moment_metrics[moment] = {
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2,
            'r': r,
            'N': len(pred)
        }
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Per-Moment Test Performance - Total_Facial_scale (Ensemble)', 
                 fontsize=16, fontweight='bold')
    
    moment_labels = [m for m in moments if m in moment_metrics]
    mae_vals = [moment_metrics[m]['MAE'] for m in moment_labels]
    rmse_vals = [moment_metrics[m]['RMSE'] for m in moment_labels]
    r2_vals = [moment_metrics[m]['R²'] for m in moment_labels]
    r_vals = [moment_metrics[m]['r'] for m in moment_labels]
    
    # MAE
    axes[0, 0].bar(moment_labels, mae_vals, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Moment', fontsize=12)
    axes[0, 0].set_ylabel('MAE', fontsize=12)
    axes[0, 0].set_title('Mean Absolute Error by Moment', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    for i, (m, v) in enumerate(zip(moment_labels, mae_vals)):
        axes[0, 0].text(i, v, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # RMSE
    axes[0, 1].bar(moment_labels, rmse_vals, color='coral', alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Moment', fontsize=12)
    axes[0, 1].set_ylabel('RMSE', fontsize=12)
    axes[0, 1].set_title('Root Mean Squared Error by Moment', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    for i, (m, v) in enumerate(zip(moment_labels, rmse_vals)):
        axes[0, 1].text(i, v, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # R²
    axes[1, 0].bar(moment_labels, r2_vals, color='mediumseagreen', alpha=0.7, edgecolor='black')
    axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1, 0].set_xlabel('Moment', fontsize=12)
    axes[1, 0].set_ylabel('R²', fontsize=12)
    axes[1, 0].set_title('R² by Moment', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    for i, (m, v) in enumerate(zip(moment_labels, r2_vals)):
        axes[1, 0].text(i, v, f'{v:.2f}', ha='center', 
                       va='bottom' if v >= 0 else 'top', fontweight='bold')
    
    # Correlation
    axes[1, 1].bar(moment_labels, r_vals, color='gold', alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Moment', fontsize=12)
    axes[1, 1].set_ylabel('Pearson r', fontsize=12)
    axes[1, 1].set_title('Pearson Correlation by Moment', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    for i, (m, v) in enumerate(zip(moment_labels, r_vals)):
        if not np.isnan(v):
            axes[1, 1].text(i, v, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def plot_per_animal_results(ensemble_predictions, ensemble_targets, ensemble_animals):
    """Plot per-animal performance"""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from scipy.stats import pearsonr
    
    animal_metrics = {}
    for animal in test_animals:
        mask = np.array(ensemble_animals) == animal
        if mask.sum() == 0:
            continue
        
        pred = ensemble_predictions['Total_Facial_scale'][mask]
        target = ensemble_targets['Total_Facial_scale'][mask]
        
        mae = mean_absolute_error(target, pred)
        rmse = np.sqrt(mean_squared_error(target, pred))
        r2 = r2_score(target, pred)
        try:
            r, _ = pearsonr(target, pred)
        except:
            r = np.nan
        
        animal_metrics[animal] = {
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2,
            'r': r,
            'N': len(pred)
        }
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Per-Animal Test Performance - Total_Facial_scale (Ensemble)', 
                 fontsize=16, fontweight='bold')
    
    animal_labels = [f'Animal {a}' for a in sorted(animal_metrics.keys())]
    mae_vals = [animal_metrics[a]['MAE'] for a in sorted(animal_metrics.keys())]
    rmse_vals = [animal_metrics[a]['RMSE'] for a in sorted(animal_metrics.keys())]
    r2_vals = [animal_metrics[a]['R²'] for a in sorted(animal_metrics.keys())]
    r_vals = [animal_metrics[a]['r'] for a in sorted(animal_metrics.keys())]
    
    # MAE
    axes[0, 0].bar(animal_labels, mae_vals, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Animal', fontsize=12)
    axes[0, 0].set_ylabel('MAE', fontsize=12)
    axes[0, 0].set_title('Mean Absolute Error by Animal', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    for i, (a, v) in enumerate(zip(animal_labels, mae_vals)):
        axes[0, 0].text(i, v, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # RMSE
    axes[0, 1].bar(animal_labels, rmse_vals, color='coral', alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Animal', fontsize=12)
    axes[0, 1].set_ylabel('RMSE', fontsize=12)
    axes[0, 1].set_title('Root Mean Squared Error by Animal', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    for i, (a, v) in enumerate(zip(animal_labels, rmse_vals)):
        axes[0, 1].text(i, v, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # R²
    axes[1, 0].bar(animal_labels, r2_vals, color='mediumseagreen', alpha=0.7, edgecolor='black')
    axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1, 0].set_xlabel('Animal', fontsize=12)
    axes[1, 0].set_ylabel('R²', fontsize=12)
    axes[1, 0].set_title('R² by Animal', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    for i, (a, v) in enumerate(zip(animal_labels, r2_vals)):
        axes[1, 0].text(i, v, f'{v:.2f}', ha='center', 
                       va='bottom' if v >= 0 else 'top', fontweight='bold')
    
    # Correlation
    axes[1, 1].bar(animal_labels, r_vals, color='gold', alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Animal', fontsize=12)
    axes[1, 1].set_ylabel('Pearson r', fontsize=12)
    axes[1, 1].set_title('Pearson Correlation by Animal', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    for i, (a, v) in enumerate(zip(animal_labels, r_vals)):
        if not np.isnan(v):
            axes[1, 1].text(i, v, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def plot_feature_wise_performance(ensemble_predictions, ensemble_targets):
    """Plot feature-wise performance"""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from scipy.stats import pearsonr
    
    features = ['Orbital_tightening', 'Tension_above_eyes', 'Cheek_tightening',
                'Ears_frontal', 'Ears_lateral', 'Lip_jaw_profile', 'Nostril_muzzle']
    
    feature_metrics = {}
    for feature in features:
        if feature not in ensemble_predictions:
            continue
        
        pred = ensemble_predictions[feature]
        target = ensemble_targets[feature]
        
        mae = mean_absolute_error(target, pred)
        rmse = np.sqrt(mean_squared_error(target, pred))
        r2 = r2_score(target, pred)
        try:
            r, _ = pearsonr(target, pred)
        except:
            r = np.nan
        
        feature_metrics[feature] = {
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2,
            'r': r
        }
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Feature-Wise Test Performance (Ensemble)', 
                 fontsize=16, fontweight='bold')
    
    feature_labels = [f.replace('_', ' ').title() for f in features if f in feature_metrics]
    mae_vals = [feature_metrics[f]['MAE'] for f in features if f in feature_metrics]
    rmse_vals = [feature_metrics[f]['RMSE'] for f in features if f in feature_metrics]
    r2_vals = [feature_metrics[f]['R²'] for f in features if f in feature_metrics]
    r_vals = [feature_metrics[f]['r'] for f in features if f in feature_metrics]
    
    # MAE
    axes[0, 0].barh(feature_labels, mae_vals, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('MAE', fontsize=12)
    axes[0, 0].set_ylabel('Feature', fontsize=12)
    axes[0, 0].set_title('Mean Absolute Error by Feature', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='x')
    for i, (f, v) in enumerate(zip(feature_labels, mae_vals)):
        axes[0, 0].text(v, i, f'{v:.3f}', ha='left', va='center', fontweight='bold')
    
    # RMSE
    axes[0, 1].barh(feature_labels, rmse_vals, color='coral', alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('RMSE', fontsize=12)
    axes[0, 1].set_ylabel('Feature', fontsize=12)
    axes[0, 1].set_title('Root Mean Squared Error by Feature', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='x')
    for i, (f, v) in enumerate(zip(feature_labels, rmse_vals)):
        axes[0, 1].text(v, i, f'{v:.3f}', ha='left', va='center', fontweight='bold')
    
    # R²
    axes[1, 0].barh(feature_labels, r2_vals, color='mediumseagreen', alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    axes[1, 0].set_xlabel('R²', fontsize=12)
    axes[1, 0].set_ylabel('Feature', fontsize=12)
    axes[1, 0].set_title('R² by Feature', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    for i, (f, v) in enumerate(zip(feature_labels, r2_vals)):
        axes[1, 0].text(v, i, f'{v:.3f}', ha='left' if v >= 0 else 'right', 
                       va='center', fontweight='bold')
    
    # Correlation
    axes[1, 1].barh(feature_labels, r_vals, color='gold', alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Pearson r', fontsize=12)
    axes[1, 1].set_ylabel('Feature', fontsize=12)
    axes[1, 1].set_title('Pearson Correlation by Feature', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    for i, (f, v) in enumerate(zip(feature_labels, r_vals)):
        if not np.isnan(v):
            axes[1, 1].text(v, i, f'{v:.3f}', ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# GENERATE ALL VISUALIZATIONS
# ============================================================================
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

print("\n📊 1. Per-Fold Results")
plot_per_fold_results(all_fold_results, ensemble_metrics)

print("\n📊 2. Per-Moment Results")
plot_per_moment_results(ensemble_predictions, ensemble_targets, ensemble_moments)

print("\n📊 3. Per-Animal Results")
plot_per_animal_results(ensemble_predictions, ensemble_targets, ensemble_animals)

print("\n📊 4. Feature-Wise Performance")
plot_feature_wise_performance(ensemble_predictions, ensemble_targets)

print("\n" + "="*80)
print("✅ ALL VISUALIZATIONS COMPLETE")
print("="*80)

