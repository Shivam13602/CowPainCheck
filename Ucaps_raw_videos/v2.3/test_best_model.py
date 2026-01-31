# ============================================================================
# TEST CURRENT BEST MODEL (Fold 7)
# Comprehensive evaluation to decide if retraining is needed
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
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# SETUP
# ============================================================================
print("="*80)
print("TESTING CURRENT BEST MODEL (Fold 7)")
print("="*80)

# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Set paths
base_path = Path('/content/drive/MyDrive')
project_dir = base_path / 'facial_pain_project_v2'
sequence_dir = base_path / 'sequence'
checkpoint_dir = project_dir / 'checkpoints'

# Find sequence root
def find_sequence_root():
    possible_paths = [
        base_path / 'sequence',
        base_path / 'facial_pain_project_v2' / 'sequence',
        base_path / 'UCAPS' / 'sequence',
    ]
    for path in possible_paths:
        if path.exists():
            return path
    return None

sequence_dir = find_sequence_root()
if sequence_dir is None:
    raise FileNotFoundError("Could not find sequence/ directory")

print(f"✅ Sequence dir: {sequence_dir}")
print(f"✅ Checkpoint dir: {checkpoint_dir}")

# ============================================================================
# MODEL ARCHITECTURE (Must match training exactly)
# ============================================================================
print("\n[1] Loading model architecture...")

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, lstm_output):
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        return context_vector, attention_weights

class TemporalPainModel_v2(nn.Module):
    """v2.0: Enhanced model with Total Facial Scale calculation - MUST MATCH TRAINING"""
    def __init__(self, num_frames=32, lstm_hidden_size=256):
        super(TemporalPainModel_v2, self).__init__()
        
        # 3D CNN
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
        x = x.permute(0, 2, 1, 3, 4)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        
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

print("✅ Model architecture loaded")

# ============================================================================
# DATASET CLASS
# ============================================================================
print("\n[2] Loading dataset class...")

class FacialPainDataset_v2(Dataset):
    def __init__(self, sequence_mapping, sequence_dir, max_frames=32, transform=None):
        self.sequence_mapping = sequence_mapping
        self.sequence_dir = Path(sequence_dir)
        self.max_frames = max_frames
        self.transform = transform
        
        self.label_name_map = {
            'Total.Facial.scale': 'Total_Facial_scale',
            '1.Orbital.tightening': 'Orbital_tightening',
            '2.Tension.above.eyes': 'Tension_above_eyes',
            '3.Cheek.(masseter.muscle).tightnening': 'Cheek_tightening',
            '4.Ears.Position.Frontal': 'Ears_frontal',
            '5.Ears.Position.Lateral': 'Ears_lateral',
            '6.Abnormal.Lip.and.Jaw.Profile': 'Lip_jaw_profile',
            '7.Abnormal.Nostril.Muzzle.Shape': 'Nostril_muzzle',
        }
    
    def __len__(self):
        return len(self.sequence_mapping)
    
    def __getitem__(self, idx):
        seq_info = self.sequence_mapping[idx]
        
        # Handle sequence path
        if 'sequence_path' in seq_info:
            seq_path = self.sequence_dir / seq_info['sequence_path']
        elif 'sequence_id' in seq_info:
            seq_path = self.sequence_dir / seq_info['sequence_id']
        else:
            raise KeyError(f"Missing sequence_path or sequence_id")
        
        frame_files = sorted(list(seq_path.glob('*.jpg')) + list(seq_path.glob('*.png')))
        if len(frame_files) == 0:
            raise FileNotFoundError(f"No frames found in {seq_path}")
        
        if len(frame_files) > self.max_frames:
            indices = np.linspace(0, len(frame_files)-1, self.max_frames, dtype=int)
            frame_files = [frame_files[i] for i in indices]
        
        frames = []
        for frame_file in frame_files:
            img = Image.open(frame_file).convert('RGB')
            if self.transform:
                img = self.transform(img)
            frames.append(img)
        
        video_tensor = torch.stack(frames)
        
        labels = {}
        csv_label_cols = [
            'Orbital_tightening', 'Tension_above_eyes', 'Cheek_tightening',
            'Ears_frontal', 'Ears_lateral', 'Lip_jaw_profile', 'Nostril_muzzle',
            'Total.Facial.scale'
        ]
        
        for csv_col in csv_label_cols:
            val = seq_info.get(csv_col, np.nan)
            if pd.isna(val):
                model_col = self.label_name_map.get(csv_col, csv_col.replace('.', '_'))
                val = seq_info.get(model_col, np.nan)
            
            model_col = self.label_name_map.get(csv_col, csv_col.replace('.', '_'))
            labels[model_col] = torch.tensor(
                val if not pd.isna(val) else 0.0, 
                dtype=torch.float32
            )
        
        moment = seq_info.get('moment', seq_info.get('Moment', 'M0'))
        animal_id = seq_info.get('animal', seq_info.get('animal_id', 'unknown'))
        
        return video_tensor, labels, moment, animal_id

print("✅ Dataset class loaded")

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[3] Loading test data...")

splits_file = project_dir / 'train_val_test_splits_v2.json'
mapping_file = project_dir / 'sequence_label_mapping_v2.json'

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

test_animals = splits.get('test_animals', [14, 17])
test_sequences = []
for seq_info in all_sequences:
    animal_id = seq_info.get('animal') or seq_info.get('animal_id')
    if animal_id in test_animals:
        if 'sequence_id' not in seq_info:
            seq_info['sequence_id'] = seq_info.get('sequence_path', f'seq_{len(test_sequences)}')
        test_sequences.append(seq_info)

print(f"✅ Test animals: {test_animals}")
print(f"✅ Test sequences: {len(test_sequences)}")

# Transforms
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_dataset = FacialPainDataset_v2(test_sequences, sequence_dir, max_frames=32, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

# ============================================================================
# LOAD BEST MODEL (Fold 7)
# ============================================================================
print("\n[4] Loading best model (Fold 7)...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Device: {device}")

model = TemporalPainModel_v2(num_frames=32).to(device)
best_model_path = checkpoint_dir / 'best_model_v2_fold_7.pt'

if not best_model_path.exists():
    raise FileNotFoundError(f"Best model not found: {best_model_path}")

checkpoint = torch.load(best_model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

epoch = checkpoint.get('epoch', 'unknown')
val_loss = checkpoint.get('val_loss', 'unknown')
print(f"✅ Model loaded (epoch {epoch}, val_loss={val_loss:.4f})")

model.eval()

# ============================================================================
# EVALUATION
# ============================================================================
print("\n[5] Running evaluation...")

all_predictions = {task: [] for task in ['Orbital_tightening', 'Tension_above_eyes', 'Cheek_tightening',
                                         'Ears_frontal', 'Ears_lateral', 'Lip_jaw_profile', 
                                         'Nostril_muzzle', 'Total_Facial_scale']}
all_targets = {task: [] for task in all_predictions.keys()}
all_moments = []
all_animals = []

with torch.no_grad():
    for frames, labels, moments, animals in tqdm(test_loader, desc="Evaluating"):
        frames = frames.to(device)
        
        outputs, _ = model(frames)
        
        for task in all_predictions.keys():
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

# ============================================================================
# CALCULATE METRICS
# ============================================================================
print("\n[6] Calculating metrics...")

results = {}
for task in all_predictions.keys():
    pred = all_predictions[task]
    target = all_targets[task]
    
    mae = mean_absolute_error(target, pred)
    rmse = np.sqrt(mean_squared_error(target, pred))
    r2 = r2_score(target, pred)
    r, p_value = pearsonr(target, pred)
    
    results[task] = {
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2,
        'r': r,
        'p_value': p_value,
        'predictions': pred,
        'targets': target
    }

# Moment-wise metrics
moment_metrics = {}
for moment in ['M0', 'M1', 'M2', 'M3', 'M4']:
    moment_mask = np.array([m == moment for m in all_moments])
    if moment_mask.sum() == 0:
        continue
    
    moment_metrics[moment] = {}
    for task in all_predictions.keys():
        pred_moment = all_predictions[task][moment_mask]
        target_moment = all_targets[task][moment_mask]
        
        mae = mean_absolute_error(target_moment, pred_moment)
        moment_metrics[moment][task] = {'MAE': mae, 'count': moment_mask.sum()}

# ============================================================================
# DISPLAY RESULTS
# ============================================================================
print("\n" + "="*80)
print("COMPREHENSIVE TEST RESULTS - FOLD 7")
print("="*80)

print("\n📊 OVERALL PERFORMANCE:")
print("-" * 80)
print(f"{'Task':<25} {'MAE':<8} {'RMSE':<8} {'R²':<10} {'r':<8} {'p-value':<10}")
print("-" * 80)
for task in all_predictions.keys():
    r = results[task]
    print(f"{task:<25} {r['MAE']:<8.3f} {r['RMSE']:<8.3f} {r['R²']:<10.3f} {r['r']:<8.3f} {r['p_value']:<10.4f}")

print("\n📊 MOMENT-WISE PERFORMANCE (Total_Facial_scale):")
print("-" * 80)
print(f"{'Moment':<10} {'MAE':<10} {'Count':<10} {'Description'}")
print("-" * 80)
moment_descriptions = {
    'M0': 'Baseline (pre-surgery)',
    'M1': 'Early post-op (~30 min)',
    'M2': 'Peak pain (~2-4 hours)',
    'M3': 'Declining (~6-8 hours)',
    'M4': 'Residual (~24 hours)'
}
for moment in ['M0', 'M1', 'M2', 'M3', 'M4']:
    if moment in moment_metrics:
        mae = moment_metrics[moment]['Total_Facial_scale']['MAE']
        count = moment_metrics[moment]['Total_Facial_scale']['count']
        desc = moment_descriptions.get(moment, '')
        print(f"{moment:<10} {mae:<10.3f} {count:<10} {desc}")

# Key metrics summary
total_facial = results['Total_Facial_scale']
print("\n" + "="*80)
print("KEY METRICS SUMMARY")
print("="*80)
print(f"Total Facial Scale:")
print(f"  • R² Score: {total_facial['R²']:.3f} ({'✅ Positive' if total_facial['R²'] > 0 else '❌ Negative'})")
print(f"  • Pearson r: {total_facial['r']:.3f} (p={total_facial['p_value']:.4f})")
print(f"  • MAE: {total_facial['MAE']:.3f} (14% error on 0-14 scale)")
print(f"  • RMSE: {total_facial['RMSE']:.3f}")

if 'M2' in moment_metrics:
    m2_mae = moment_metrics['M2']['Total_Facial_scale']['MAE']
    m0_mae = moment_metrics['M0']['Total_Facial_scale']['MAE']
    ratio = m2_mae / m0_mae if m0_mae > 0 else 0
    print(f"\nM2 (Peak Pain) Challenge:")
    print(f"  • M2 MAE: {m2_mae:.3f}")
    print(f"  • M0 MAE: {m0_mae:.3f}")
    print(f"  • Ratio: {ratio:.1f}× worse than baseline")

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n[7] Generating visualizations...")

# Figure 1: Overall Performance
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Model Performance Summary (Fold 7)', fontsize=16, fontweight='bold')

# MAE by task
ax = axes[0, 0]
tasks = list(all_predictions.keys())
mae_values = [results[t]['MAE'] for t in tasks]
bars = ax.barh(tasks, mae_values, color='steelblue', edgecolor='black')
ax.set_xlabel('Mean Absolute Error (MAE)', fontweight='bold')
ax.set_title('MAE by Task', fontweight='bold')
ax.grid(axis='x', alpha=0.3)
for i, (task, mae) in enumerate(zip(tasks, mae_values)):
    ax.text(mae + 0.05, i, f'{mae:.3f}', va='center', fontweight='bold')

# R² by task
ax = axes[0, 1]
r2_values = [results[t]['R²'] for t in tasks]
colors = ['green' if r > 0 else 'red' for r in r2_values]
bars = ax.barh(tasks, r2_values, color=colors, edgecolor='black')
ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax.set_xlabel('R² Score', fontweight='bold')
ax.set_title('R² by Task', fontweight='bold')
ax.grid(axis='x', alpha=0.3)
for i, (task, r2) in enumerate(zip(tasks, r2_values)):
    ax.text(r2 + 0.01 if r2 >= 0 else r2 - 0.01, i, f'{r2:.3f}', 
           va='center', ha='left' if r2 >= 0 else 'right', fontweight='bold')

# Moment-wise MAE
ax = axes[1, 0]
moments = ['M0', 'M1', 'M2', 'M3', 'M4']
moment_mae = [moment_metrics[m]['Total_Facial_scale']['MAE'] if m in moment_metrics else 0 for m in moments]
colors_moment = ['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c', '#9467bd']
bars = ax.bar(moments, moment_mae, color=colors_moment, edgecolor='black', linewidth=2)
bars[2].set_edgecolor('red')
bars[2].set_linewidth(3)
ax.set_ylabel('MAE', fontweight='bold')
ax.set_xlabel('Pain Assessment Moment', fontweight='bold')
ax.set_title('Total Facial Scale: Moment-wise Error\n(Red = M2 Peak Pain)', fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for bar, mae in zip(bars, moment_mae):
    if mae > 0:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
               f'{mae:.2f}', ha='center', va='bottom', fontweight='bold')

# Predicted vs Actual (Total_Facial_scale)
ax = axes[1, 1]
pred = results['Total_Facial_scale']['predictions']
target = results['Total_Facial_scale']['targets']
ax.scatter(target, pred, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
min_val = min(target.min(), pred.min())
max_val = max(target.max(), pred.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
ax.set_xlabel('Actual Total Facial Scale', fontweight='bold')
ax.set_ylabel('Predicted Total Facial Scale', fontweight='bold')
ax.set_title(f'Predicted vs Actual\n(R²={total_facial["R²"]:.3f}, r={total_facial["r"]:.3f})', fontweight='bold')
ax.grid(alpha=0.3)
ax.legend()

plt.tight_layout()
plt.show()

# Figure 2: Task-wise correlations
fig, ax = plt.subplots(figsize=(12, 6))
tasks_short = [t.replace('_', ' ').title() for t in tasks]
r_values = [results[t]['r'] for t in tasks]
colors_r = ['green' if r > 0.3 else 'orange' if r > 0 else 'red' for r in r_values]
bars = ax.barh(tasks_short, r_values, color=colors_r, edgecolor='black')
ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax.set_xlabel('Pearson Correlation (r)', fontweight='bold', fontsize=12)
ax.set_title('Task-wise Pearson Correlation', fontweight='bold', fontsize=14)
ax.grid(axis='x', alpha=0.3)
for i, (task, r) in enumerate(zip(tasks_short, r_values)):
    ax.text(r + 0.02 if r >= 0 else r - 0.02, i, f'{r:.3f}', 
           va='center', ha='left' if r >= 0 else 'right', fontweight='bold')
plt.tight_layout()
plt.show()

# ============================================================================
# ASSESSMENT
# ============================================================================
print("\n" + "="*80)
print("MODEL ASSESSMENT")
print("="*80)

assessment = []

# R² check
if total_facial['R²'] > 0.15:
    assessment.append("✅ R² > 0.15: Good model performance")
elif total_facial['R²'] > 0.05:
    assessment.append("⚠️  R² 0.05-0.15: Moderate performance, room for improvement")
else:
    assessment.append("❌ R² < 0.05: Poor performance, retraining recommended")

# Correlation check
if total_facial['r'] > 0.5:
    assessment.append("✅ Strong correlation (r > 0.5)")
elif total_facial['r'] > 0.3:
    assessment.append("⚠️  Moderate correlation (r 0.3-0.5)")
else:
    assessment.append("❌ Weak correlation (r < 0.3)")

# MAE check
if total_facial['MAE'] < 1.5:
    assessment.append("✅ Low MAE (< 1.5): Excellent accuracy")
elif total_facial['MAE'] < 2.0:
    assessment.append("⚠️  Moderate MAE (1.5-2.0): Acceptable accuracy")
else:
    assessment.append("❌ High MAE (> 2.0): Poor accuracy")

# M2 check
if 'M2' in moment_metrics:
    m2_mae = moment_metrics['M2']['Total_Facial_scale']['MAE']
    if m2_mae < 2.5:
        assessment.append("✅ M2 MAE < 2.5: Good acute pain detection")
    elif m2_mae < 3.5:
        assessment.append("⚠️  M2 MAE 2.5-3.5: Moderate acute pain detection")
    else:
        assessment.append("❌ M2 MAE > 3.5: Poor acute pain detection - RETRAINING RECOMMENDED")

for item in assessment:
    print(f"  {item}")

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)

# Calculate recommendation score
score = 0
if total_facial['R²'] > 0.15:
    score += 2
elif total_facial['R²'] > 0.05:
    score += 1

if total_facial['r'] > 0.5:
    score += 2
elif total_facial['r'] > 0.3:
    score += 1

if total_facial['MAE'] < 1.5:
    score += 2
elif total_facial['MAE'] < 2.0:
    score += 1

if 'M2' in moment_metrics:
    m2_mae = moment_metrics['M2']['Total_Facial_scale']['MAE']
    if m2_mae < 2.5:
        score += 2
    elif m2_mae < 3.5:
        score += 1

if score >= 7:
    recommendation = "✅ MODEL IS ACCEPTABLE - Can be used for deployment"
    action = "Deploy this model. Consider fine-tuning for M2 if needed."
elif score >= 4:
    recommendation = "⚠️  MODEL IS MARGINAL - Consider retraining"
    action = "Test on real-world data first. If acceptable, use it. Otherwise, retrain with improved hyperparameters."
else:
    recommendation = "❌ MODEL NEEDS RETRAINING - Performance is insufficient"
    action = "Retrain all folds with improved hyperparameters (lower LR, higher weight decay, better augmentation, fixed early stopping)."

print(f"\n{recommendation}")
print(f"\nAction: {action}")
print(f"\nScore: {score}/8 (7+ = Acceptable, 4-6 = Marginal, <4 = Needs Retraining)")

print("\n" + "="*80)
print("✅ TEST COMPLETE")
print("="*80)

