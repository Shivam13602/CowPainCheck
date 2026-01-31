# ============================================================================
# TEST SET EVALUATION - v2.4 (Regression Only)
# Evaluates models on held-out test set (animals 14 and 17)
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
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import os
warnings.filterwarnings('ignore')

# ============================================================================
# SETUP
# ============================================================================
print("="*80)
print("TEST SET EVALUATION - v2.4 (Regression Only)")
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
print(f"\n📊 Test Animals: {test_animals}")

# Get test sequences
test_sequences = [s for s in all_sequences if s.get('animal', s.get('animal_id')) in test_animals]
print(f"📊 Test Sequences: {len(test_sequences)}")

if len(test_sequences) == 0:
    print("⚠️  No test sequences found!")
    exit(1)

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

        print("   Pre-computing frame paths...")
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
def evaluate_model(model, dataloader, device, use_ensemble=False):
    """Evaluate model(s) on test set"""
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
# MAIN EVALUATION
# ============================================================================
print("\n" + "="*80)
print("EVALUATING ON TEST SET")
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

print(f"\n📊 Test Set: {len(test_sequences)} sequences")
print(f"   Animals: {test_animals}")

# Evaluate with each fold model
print("\n" + "="*80)
print("EVALUATING WITH INDIVIDUAL FOLD MODELS")
print("="*80)

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
    
    if 'Total_Facial_scale' in metrics:
        m = metrics['Total_Facial_scale']
        print(f"   ✅ MAE: {m['MAE']:.4f}, RMSE: {m['RMSE']:.4f}, R²: {m['R²']:.4f}, r: {m['r']:.4f}")

if len(all_fold_results) == 0:
    print("\n⚠️  No models found for evaluation!")
    exit(1)

# Ensemble predictions (average across folds)
print("\n" + "="*80)
print("ENSEMBLE EVALUATION (Average across all folds)")
print("="*80)

ensemble_predictions = defaultdict(list)
ensemble_targets = defaultdict(list)
ensemble_moments = []
ensemble_animals = []

# Collect all predictions
for result in all_fold_results:
    for task in result['predictions'].keys():
        ensemble_predictions[task].append(result['predictions'][task])
    ensemble_targets = result['targets']  # Same for all
    ensemble_moments = result['moments']  # Same for all
    ensemble_animals = result['animals']  # Same for all

# Average predictions
for task in ensemble_predictions.keys():
    ensemble_predictions[task] = np.mean(ensemble_predictions[task], axis=0)

ensemble_metrics = compute_metrics(ensemble_predictions, ensemble_targets)

# Display results
print("\n📊 ENSEMBLE RESULTS - Total_Facial_scale:")
print("-" * 80)
if 'Total_Facial_scale' in ensemble_metrics:
    m = ensemble_metrics['Total_Facial_scale']
    print(f"MAE:  {m['MAE']:.4f}")
    print(f"RMSE: {m['RMSE']:.4f}")
    print(f"R²:   {m['R²']:.4f}")
    print(f"r:    {m['r']:.4f} (p={m['p_value']:.4f})")
    print(f"N:    {m['N']}")

# Per-fold summary
print("\n" + "="*80)
print("PER-FOLD TEST RESULTS - Total_Facial_scale")
print("="*80)

fold_summary = []
for result in all_fold_results:
    if 'Total_Facial_scale' in result['metrics']:
        m = result['metrics']['Total_Facial_scale']
        fold_summary.append({
            'Fold': result['fold'],
            'MAE': m['MAE'],
            'RMSE': m['RMSE'],
            'R²': m['R²'],
            'r': m['r'],
            'N': m['N']
        })

if fold_summary:
    df = pd.DataFrame(fold_summary)
    print("\n📊 Per-Fold Performance:")
    print(df.to_string(index=False))
    
    print("\n📊 Aggregate Statistics:")
    print("-" * 80)
    print(f"{'Metric':<15} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print("-" * 80)
    print(f"{'MAE':<15} {df['MAE'].mean():<12.4f} {df['MAE'].std():<12.4f} {df['MAE'].min():<12.4f} {df['MAE'].max():<12.4f}")
    print(f"{'RMSE':<15} {df['RMSE'].mean():<12.4f} {df['RMSE'].std():<12.4f} {df['RMSE'].min():<12.4f} {df['RMSE'].max():<12.4f}")
    print(f"{'R²':<15} {df['R²'].mean():<12.4f} {df['R²'].std():<12.4f} {df['R²'].min():<12.4f} {df['R²'].max():<12.4f}")
    print(f"{'r (correlation)':<15} {df['r'].mean():<12.4f} {df['r'].std():<12.4f} {df['r'].min():<12.4f} {df['r'].max():<12.4f}")

# Moment-wise analysis
print("\n" + "="*80)
print("MOMENT-WISE TEST RESULTS - Total_Facial_scale (Ensemble)")
print("="*80)

moment_metrics = {}
if 'Total_Facial_scale' in ensemble_predictions:
    for moment in ['M0', 'M1', 'M2', 'M3', 'M4']:
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

if moment_metrics:
    print("\n📊 Moment-Wise Performance:")
    print("-" * 80)
    print(f"{'Moment':<10} {'MAE':<12} {'RMSE':<12} {'R²':<12} {'r':<12} {'N':<6}")
    print("-" * 80)
    for moment in ['M0', 'M1', 'M2', 'M3', 'M4']:
        if moment in moment_metrics:
            m = moment_metrics[moment]
            print(f"{moment:<10} {m['MAE']:<12.4f} {m['RMSE']:<12.4f} {m['R²']:<12.4f} {m['r']:<12.4f} {m['N']:<6}")

# Individual feature performance
print("\n" + "="*80)
print("INDIVIDUAL FEATURE PERFORMANCE (Ensemble)")
print("="*80)

feature_metrics = {}
for feature in ['Orbital_tightening', 'Tension_above_eyes', 'Cheek_tightening',
                'Ears_frontal', 'Ears_lateral', 'Lip_jaw_profile', 'Nostril_muzzle']:
    if feature in ensemble_predictions:
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

if feature_metrics:
    print("\n📊 Feature-Wise Performance:")
    print("-" * 80)
    print(f"{'Feature':<25} {'MAE':<12} {'RMSE':<12} {'R²':<12} {'r':<12}")
    print("-" * 80)
    for feature in sorted(feature_metrics.keys()):
        m = feature_metrics[feature]
        print(f"{feature:<25} {m['MAE']:<12.4f} {m['RMSE']:<12.4f} {m['R²']:<12.4f} {m['r']:<12.4f}")

# Animal-wise analysis
print("\n" + "="*80)
print("ANIMAL-WISE TEST RESULTS - Total_Facial_scale (Ensemble)")
print("="*80)

animal_metrics = {}
if 'Total_Facial_scale' in ensemble_predictions:
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

if animal_metrics:
    print("\n📊 Animal-Wise Performance:")
    print("-" * 80)
    print(f"{'Animal':<10} {'MAE':<12} {'RMSE':<12} {'R²':<12} {'r':<12} {'N':<6}")
    print("-" * 80)
    for animal in sorted(animal_metrics.keys()):
        m = animal_metrics[animal]
        print(f"{animal:<10} {m['MAE']:<12.4f} {m['RMSE']:<12.4f} {m['R²']:<12.4f} {m['r']:<12.4f} {m['N']:<6}")

print("\n" + "="*80)
print("✅ TEST EVALUATION COMPLETE")
print("="*80)

