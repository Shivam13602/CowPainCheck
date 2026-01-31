# ============================================================================
# VALIDATION RESULTS ANALYZER - v2.4 (Regression Only)
# Analyzes validation results across all folds and prepares for evaluation
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
warnings.filterwarnings('ignore')

# ============================================================================
# SETUP
# ============================================================================
print("="*80)
print("VALIDATION RESULTS ANALYZER - v2.4 (Regression Only)")
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
# DATASET CLASS (Same as training)
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
def evaluate_fold(fold_idx, checkpoint_dir, sequence_dir, all_sequences, splits, device):
    """Evaluate a single fold on its validation set"""
    
    # Get fold data
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
    
    # Load best model
    best_model_path = checkpoint_dir / f'best_model_v2.4_fold_{fold_idx}.pt'
    if not best_model_path.exists():
        print(f"   ⚠️  Best model not found for fold {fold_idx}")
        return None
    
    # Load checkpoint
    checkpoint = torch.load(best_model_path, map_location=device)
    val_loss = checkpoint.get('val_loss', None)
    epoch = checkpoint.get('epoch', None)
    
    # Create dataset and dataloader
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_dataset = FacialPainDataset_v2_4(val_sequences, sequence_dir, max_frames=32, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # Load model
    model = TemporalPainModel_v2_4(num_frames=32, lstm_hidden_size=128, use_bidirectional=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Collect predictions
    all_predictions = defaultdict(list)
    all_targets = defaultdict(list)
    all_moments = []
    
    with torch.no_grad():
        for frames, labels, metadata in val_loader:
            frames = frames.to(device)
            moments = metadata['moment']
            
            outputs, _ = model(frames)
            
            for task in outputs.keys():
                if task in ['Total_Facial_scale_calculated', 'Total_Facial_scale_predicted']:
                    continue
                pred = outputs[task].cpu().numpy()
                target = labels[task].cpu().numpy()
                all_predictions[task].extend(pred)
                all_targets[task].extend(target)
            
            all_moments.extend(moments)
    
    # Convert to numpy
    for task in all_predictions.keys():
        all_predictions[task] = np.array(all_predictions[task])
        all_targets[task] = np.array(all_targets[task])
    
    # Compute metrics
    results = {}
    for task in all_predictions.keys():
        pred = all_predictions[task]
        target = all_targets[task]
        
        mae = mean_absolute_error(target, pred)
        rmse = np.sqrt(mean_squared_error(target, pred))
        r2 = r2_score(target, pred)
        r, p = pearsonr(target, pred)
        
        results[task] = {
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2,
            'r': r,
            'p_value': p,
            'N': len(pred),
            'val_loss': val_loss,
            'epoch': epoch
        }
    
    # Moment-wise metrics for Total_Facial_scale
    moment_metrics = {}
    if 'Total_Facial_scale' in all_predictions:
        for moment in ['M0', 'M1', 'M2', 'M3', 'M4']:
            mask = np.array(all_moments) == moment
            if mask.sum() == 0:
                continue
            
            pred = all_predictions['Total_Facial_scale'][mask]
            target = all_targets['Total_Facial_scale'][mask]
            
            mae = mean_absolute_error(target, pred)
            rmse = np.sqrt(mean_squared_error(target, pred))
            r2 = r2_score(target, pred)
            r, _ = pearsonr(target, pred)
            
            moment_metrics[moment] = {
                'MAE': mae,
                'RMSE': rmse,
                'R²': r2,
                'r': r,
                'N': len(pred)
            }
    
    return {
        'fold': fold_idx,
        'val_animals': val_animals,
        'n_sequences': len(val_sequences),
        'results': results,
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

# Extract validation losses from training output
training_val_losses = {
    0: 1.2708,
    1: 3.3584,
    2: 1.9351,
    3: 2.1420,
    4: 4.3815,
    5: 7.6223,
    6: 8.1645,
    7: 1.9867,
    8: 1.4094
}

print("\n📊 VALIDATION LOSS SUMMARY (from training output):")
print("-" * 80)
print(f"{'Fold':<6} {'Val Loss':<12} {'Status'}")
print("-" * 80)
for fold_idx in range(9):
    val_loss = training_val_losses.get(fold_idx, None)
    if val_loss is not None:
        if val_loss < 2.0:
            status = "✅ Excellent"
        elif val_loss < 3.0:
            status = "✅ Good"
        elif val_loss < 5.0:
            status = "⚠️  Moderate"
        else:
            status = "❌ Poor"
        print(f"{fold_idx:<6} {val_loss:<12.4f} {status}")

mean_val_loss = np.mean(list(training_val_losses.values()))
std_val_loss = np.std(list(training_val_losses.values()))
min_val_loss = min(training_val_losses.values())
max_val_loss = max(training_val_losses.values())

print("-" * 80)
print(f"{'Mean':<6} {mean_val_loss:<12.4f}")
print(f"{'Std':<6} {std_val_loss:<12.4f}")
print(f"{'Min':<6} {min_val_loss:<12.4f} (Fold {min(training_val_losses, key=training_val_losses.get)})")
print(f"{'Max':<6} {max_val_loss:<12.4f} (Fold {max(training_val_losses, key=training_val_losses.get)})")

# Evaluate each fold
print("\n" + "="*80)
print("EVALUATING MODELS ON VALIDATION SETS")
print("="*80)

all_fold_results = []
for fold_idx in range(9):
    print(f"\n📊 Evaluating Fold {fold_idx}...")
    try:
        result = evaluate_fold(fold_idx, checkpoint_dir, sequence_dir, all_sequences, splits, device)
        if result:
            all_fold_results.append(result)
            print(f"   ✅ Fold {fold_idx}: {result['n_sequences']} sequences evaluated")
        else:
            print(f"   ⚠️  Fold {fold_idx}: Could not evaluate")
    except Exception as e:
        print(f"   ❌ Fold {fold_idx}: Error - {e}")

if len(all_fold_results) == 0:
    print("\n⚠️  No fold results available. Check checkpoint paths.")
else:
    # Aggregate results
    print("\n" + "="*80)
    print("AGGREGATE VALIDATION RESULTS - Total_Facial_scale")
    print("="*80)
    
    # Collect all metrics
    all_mae = []
    all_rmse = []
    all_r2 = []
    all_r = []
    
    fold_summary = []
    for result in all_fold_results:
        if 'Total_Facial_scale' in result['results']:
            metrics = result['results']['Total_Facial_scale']
            all_mae.append(metrics['MAE'])
            all_rmse.append(metrics['RMSE'])
            all_r2.append(metrics['R²'])
            all_r.append(metrics['r'])
            
            fold_summary.append({
                'Fold': result['fold'],
                'N': metrics['N'],
                'MAE': metrics['MAE'],
                'RMSE': metrics['RMSE'],
                'R²': metrics['R²'],
                'r': metrics['r'],
                'p_value': metrics['p_value'],
                'Val_Loss': metrics['val_loss']
            })
    
    # Display summary table
    if fold_summary:
        df = pd.DataFrame(fold_summary)
        print("\n📊 Per-Fold Performance:")
        print(df.to_string(index=False))
        
        print("\n📊 Aggregate Statistics:")
        print("-" * 80)
        print(f"{'Metric':<15} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
        print("-" * 80)
        print(f"{'MAE':<15} {np.mean(all_mae):<12.4f} {np.std(all_mae):<12.4f} {np.min(all_mae):<12.4f} {np.max(all_mae):<12.4f}")
        print(f"{'RMSE':<15} {np.mean(all_rmse):<12.4f} {np.std(all_rmse):<12.4f} {np.min(all_rmse):<12.4f} {np.max(all_rmse):<12.4f}")
        print(f"{'R²':<15} {np.mean(all_r2):<12.4f} {np.std(all_r2):<12.4f} {np.min(all_r2):<12.4f} {np.max(all_r2):<12.4f}")
        print(f"{'r (correlation)':<15} {np.mean(all_r):<12.4f} {np.std(all_r):<12.4f} {np.min(all_r):<12.4f} {np.max(all_r):<12.4f}")
        
        # Moment-wise aggregate
        print("\n" + "="*80)
        print("MOMENT-WISE AGGREGATE RESULTS - Total_Facial_scale")
        print("="*80)
        
        moment_aggregates = defaultdict(lambda: {'MAE': [], 'R²': [], 'r': [], 'N': []})
        for result in all_fold_results:
            if 'moment_metrics' in result:
                for moment, metrics in result['moment_metrics'].items():
                    moment_aggregates[moment]['MAE'].append(metrics['MAE'])
                    moment_aggregates[moment]['R²'].append(metrics['R²'])
                    moment_aggregates[moment]['r'].append(metrics['r'])
                    moment_aggregates[moment]['N'].append(metrics['N'])
        
        print(f"\n{'Moment':<10} {'Mean MAE':<12} {'Mean R²':<12} {'Mean r':<12} {'Total N':<10}")
        print("-" * 80)
        for moment in ['M0', 'M1', 'M2', 'M3', 'M4']:
            if moment in moment_aggregates:
                agg = moment_aggregates[moment]
                mean_mae = np.mean(agg['MAE'])
                mean_r2 = np.mean(agg['R²'])
                mean_r = np.mean(agg['r'])
                total_n = sum(agg['N'])
                print(f"{moment:<10} {mean_mae:<12.4f} {mean_r2:<12.4f} {mean_r:<12.4f} {total_n:<10}")
        
        # Individual feature performance
        print("\n" + "="*80)
        print("INDIVIDUAL FEATURE PERFORMANCE (Aggregate across folds)")
        print("="*80)
        
        feature_aggregates = defaultdict(lambda: {'MAE': [], 'R²': [], 'r': []})
        for result in all_fold_results:
            for task, metrics in result['results'].items():
                if task != 'Total_Facial_scale':
                    feature_aggregates[task]['MAE'].append(metrics['MAE'])
                    feature_aggregates[task]['R²'].append(metrics['R²'])
                    feature_aggregates[task]['r'].append(metrics['r'])
        
        print(f"\n{'Feature':<25} {'Mean MAE':<12} {'Mean R²':<12} {'Mean r':<12}")
        print("-" * 80)
        for feature in sorted(feature_aggregates.keys()):
            agg = feature_aggregates[feature]
            mean_mae = np.mean(agg['MAE'])
            mean_r2 = np.mean(agg['R²'])
            mean_r = np.mean(agg['r'])
            print(f"{feature:<25} {mean_mae:<12.4f} {mean_r2:<12.4f} {mean_r:<12.4f}")

print("\n" + "="*80)
print("✅ VALIDATION ANALYSIS COMPLETE")
print("="*80)
print("\n💡 Next Step: Run comprehensive evaluation script on test sets")
print("   Use: evaluate_model_v2.4.py (to be created)")

