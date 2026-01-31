# ============================================================================
# MODEL EVALUATION SCRIPT (v2.0)
# Evaluates trained model on test set (Animals 14, 17)
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
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# SETUP: Mount Drive and Verify Paths
# ============================================================================
print("="*80)
print("MODEL EVALUATION (v2.0)")
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
# 1. MODEL ARCHITECTURE (Must match training exactly)
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
        
        # Output heads
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
        
        outputs = {}
        for task, head in self.output_heads.items():
            outputs[task] = head(context).squeeze(-1)
        
        individual_features = [
            outputs['Orbital_tightening'], outputs['Tension_above_eyes'],
            outputs['Cheek_tightening'], outputs['Ears_frontal'],
            outputs['Ears_lateral'], outputs['Lip_jaw_profile'],
            outputs['Nostril_muzzle']
        ]
        outputs['Total_Facial_scale_calculated'] = torch.stack(individual_features, dim=0).sum(dim=0)
        outputs['Total_Facial_scale_predicted'] = self.total_head(context).squeeze(-1)
        outputs['Total_Facial_scale'] = outputs['Total_Facial_scale_calculated']
        
        return outputs, attention_weights

print("✅ Model architecture loaded")

# ============================================================================
# 2. DATASET CLASS
# ============================================================================
print("\n[2] Loading dataset class...")

class FacialPainDataset_v2(Dataset):
    def __init__(self, sequence_mapping, sequence_dir, max_frames=32, transform=None):
        self.sequence_mapping = sequence_mapping
        self.sequence_dir = Path(sequence_dir)
        self.max_frames = max_frames
        self.transform = transform
        
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
    
    def __getitem__(self, idx):
        seq_info = self.sequence_mapping[idx]
        
        if 'sequence_path' in seq_info:
            # Handle Windows-style paths (backslashes) in JSON
            path_str = seq_info['sequence_path'].replace('\\', '/')
            seq_path = self.sequence_dir / path_str
        elif 'sequence_id' in seq_info:
            seq_path = self.sequence_dir / seq_info['sequence_id']
        else:
            raise KeyError(f"Missing sequence_path in sequence {idx}")
        
        frame_files = sorted(list(seq_path.glob('*.jpg')) + list(seq_path.glob('*.png')))
        if len(frame_files) == 0:
            raise FileNotFoundError(f"No frames found in {seq_path}")
        
        # Sample frames
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
        
        # Labels - handle both dot and underscore formats
        labels = {}
        csv_label_cols = [
            'Orbital_tightening', 'Tension_above_eyes', 'Cheek_tightening',
            'Ears_frontal', 'Ears_lateral', 'Lip_jaw_profile', 'Nostril_muzzle',
            'Total.Facial.scale'  # CSV format (dots)
        ]
        
        for csv_col in csv_label_cols:
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
        
        # Get moment and animal_id
        moment = seq_info.get('moment', seq_info.get('Moment', 'M0'))
        animal_id = seq_info.get('animal', seq_info.get('animal_id', 'unknown'))
        
        return video_tensor, labels, moment, animal_id

print("✅ Dataset class loaded")

# ============================================================================
# 3. LOAD DATA SPLITS
# ============================================================================
print("\n[3] Loading data splits...")

# Load splits
splits_file = project_dir / 'train_val_test_splits_v2.json'
mapping_file = project_dir / 'sequence_label_mapping_v2.json'

if not splits_file.exists() or not mapping_file.exists():
    raise FileNotFoundError(f"Missing split files: {splits_file} or {mapping_file}")

with open(splits_file, 'r') as f:
    splits = json.load(f)

with open(mapping_file, 'r') as f:
    sequence_mapping = json.load(f)

# Handle both list and dict formats (same as training script)
if isinstance(sequence_mapping, dict):
    if 'sequences' in sequence_mapping:
        all_sequences = sequence_mapping['sequences']
    else:
        # Convert dict to list
        all_sequences = [{'sequence_id': k, **v} for k, v in sequence_mapping.items()]
else:
    all_sequences = sequence_mapping

# Get test animals
test_animals = splits.get('test_animals', [14, 17])
print(f"✅ Test animals: {test_animals}")

# Filter test sequences
test_sequences = []
for seq_info in all_sequences:
    # Handle both 'animal' and 'animal_id' field names
    animal_id = seq_info.get('animal') or seq_info.get('animal_id')
    if animal_id in test_animals:
        # Ensure sequence_id exists
        if 'sequence_id' not in seq_info:
            seq_info['sequence_id'] = seq_info.get('sequence_path', f'seq_{len(test_sequences)}')
        test_sequences.append(seq_info)

print(f"✅ Test sequences: {len(test_sequences)}")

# ============================================================================
# 4. FIND ALL TRAINED MODELS
# ============================================================================
print("\n[4] Finding all trained models...")

# Find all best model checkpoints
all_checkpoints = sorted(checkpoint_dir.glob('best_model_v2_fold_*.pt'))
if not all_checkpoints:
    raise FileNotFoundError(f"No best_model_v2_fold_*.pt found in {checkpoint_dir}")

print(f"✅ Found {len(all_checkpoints)} trained fold(s):")
for cp in all_checkpoints:
    fold_num = int(cp.stem.split('_')[-1])
    print(f"   - Fold {fold_num}: {cp.name}")

# Ask which folds to evaluate
print("\n" + "="*80)
print("EVALUATION OPTIONS:")
print("="*80)
print("1. Evaluate all folds (recommended)")
print("2. Evaluate specific fold(s)")
print("3. Evaluate single fold (interactive)")

choice = input("\nEnter choice (1/2/3) or press Enter for option 1: ").strip()
if not choice:
    choice = "1"

if choice == "1":
    folds_to_evaluate = [int(cp.stem.split('_')[-1]) for cp in all_checkpoints]
    print(f"\n✅ Will evaluate all {len(folds_to_evaluate)} fold(s): {folds_to_evaluate}")
elif choice == "2":
    fold_input = input("Enter fold numbers (comma-separated, e.g., 3,8): ").strip()
    folds_to_evaluate = [int(f.strip()) for f in fold_input.split(',')]
    print(f"\n✅ Will evaluate fold(s): {folds_to_evaluate}")
else:
    fold_num = int(input("Enter fold number: ").strip())
    folds_to_evaluate = [fold_num]
    print(f"\n✅ Will evaluate fold: {fold_num}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n✅ Device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

# ============================================================================
# 5. EVALUATION FUNCTION
# ============================================================================
print("\n[5] Setting up evaluation...")

transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_dataset = FacialPainDataset_v2(test_sequences, sequence_dir, max_frames=32, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

def evaluate_single_fold(fold_num, checkpoint_path, device):
    """Evaluate a single fold model"""
    print(f"\n{'='*80}")
    print(f"EVALUATING FOLD {fold_num}")
    print(f"{'='*80}")
    print(f"Loading: {checkpoint_path.name}")
    
    # Load model
    model = TemporalPainModel_v2(num_frames=32).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    epoch = checkpoint.get('epoch', 'unknown')
    val_loss = checkpoint.get('val_loss', 'unknown')
    
    # Format val_loss for display
    if isinstance(val_loss, float):
        val_loss_str = f"{val_loss:.4f}"
    else:
        val_loss_str = str(val_loss)
    
    print(f"✅ Model loaded (epoch {epoch}, val_loss={val_loss_str})")
    
    # Task names (7 features + Total)
    task_names = [
        'Orbital_tightening', 'Tension_above_eyes', 'Cheek_tightening',
        'Ears_frontal', 'Ears_lateral', 'Lip_jaw_profile', 'Nostril_muzzle',
        'Total_Facial_scale'
    ]
    
    all_predictions = {task: [] for task in task_names}
    all_targets = {task: [] for task in task_names}
    all_moments = []
    all_animal_ids = []
    
    # Run inference
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(test_loader, desc=f"Fold {fold_num}")):
            videos, labels_dict, moments_list, animal_ids_list = batch_data
            videos = videos.to(device)
            
            outputs, _ = model(videos)
            
            batch_size = videos.size(0)
            for i in range(batch_size):
                for task in task_names:
                    pred = outputs[task][i].cpu()
                    if pred.dim() == 0:
                        pred = pred.item()
                    else:
                        pred = pred.numpy().item() if pred.numel() == 1 else pred.numpy()[0]
                    
                    target = labels_dict[task][i]
                    if isinstance(target, torch.Tensor):
                        target = target.item()
                    else:
                        target = float(target)
                    
                    all_predictions[task].append(pred)
                    all_targets[task].append(target)
                
                all_moments.append(str(moments_list[i]))
                all_animal_ids.append(str(animal_ids_list[i]))
    
    # Convert to numpy
    for task in task_names:
        all_predictions[task] = np.array(all_predictions[task])
        all_targets[task] = np.array(all_targets[task])
    
    # Compute metrics
    results = []
    for task in task_names:
        pred = all_predictions[task]
        target = all_targets[task]
        
        mae = mean_absolute_error(target, pred)
        rmse = np.sqrt(mean_squared_error(target, pred))
        r2 = r2_score(target, pred)
        r, p = pearsonr(target, pred)
        
        results.append({
            'Fold': fold_num,
            'Task': task,
            'N': len(pred),
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2,
            'r': r,
            'p-value': p,
            'Epoch': epoch,
            'Val_Loss': val_loss if isinstance(val_loss, float) else None
        })
    
    # Moment-wise metrics
    moment_results = []
    for task in task_names:
        for moment in ['M0', 'M1', 'M2', 'M3', 'M4']:
            mask = np.array(all_moments) == moment
            if mask.sum() == 0:
                continue
            
            pred = all_predictions[task][mask]
            target = all_targets[task][mask]
            
            mae = mean_absolute_error(target, pred)
            rmse = np.sqrt(mean_squared_error(target, pred))
            r2 = r2_score(target, pred)
            r, _ = pearsonr(target, pred)
            
            moment_results.append({
                'Fold': fold_num,
                'Task': task,
                'Moment': moment,
                'N': len(pred),
                'MAE': mae,
                'RMSE': rmse,
                'R²': r2,
                'r': r
            })
    
    return results, moment_results

# ============================================================================
# 6. EVALUATE ALL FOLDS
# ============================================================================
print("\n[6] Evaluating all folds...")

all_results = []
all_moment_results = []

for fold_num in folds_to_evaluate:
    checkpoint_path = checkpoint_dir / f'best_model_v2_fold_{fold_num}.pt'
    
    if not checkpoint_path.exists():
        print(f"⚠️  Fold {fold_num} checkpoint not found, skipping...")
        continue
    
    try:
        results, moment_results = evaluate_single_fold(fold_num, checkpoint_path, device)
        all_results.extend(results)
        all_moment_results.extend(moment_results)
    except Exception as e:
        print(f"❌ Error evaluating Fold {fold_num}: {e}")
        import traceback
        traceback.print_exc()
        continue

if not all_results:
    raise ValueError("No results collected! Check that checkpoints exist.")

# ============================================================================
# 7. GENERATE COMPREHENSIVE REPORT
# ============================================================================
print("\n" + "="*80)
print("COMPREHENSIVE EVALUATION REPORT")
print("="*80)

# Overall results by fold
results_df = pd.DataFrame(all_results)
print("\n" + "="*80)
print("OVERALL RESULTS BY FOLD (Test: Animals 14, 17)")
print("="*80)
print(results_df.to_string(index=False))

# Summary: Best fold per task
print("\n" + "="*80)
print("BEST FOLD PER TASK (by R²)")
print("="*80)
for task in results_df['Task'].unique():
    task_df = results_df[results_df['Task'] == task]
    best = task_df.loc[task_df['R²'].idxmax()]
    
    # Format val_loss for display
    if pd.notna(best['Val_Loss']) and best['Val_Loss'] is not None:
        val_loss_str = f"{best['Val_Loss']:.4f}"
    else:
        val_loss_str = 'N/A'
    
    print(f"\n{task}:")
    print(f"  Best Fold: {int(best['Fold'])} (R²={best['R²']:.3f}, r={best['r']:.3f}, MAE={best['MAE']:.3f})")
    print(f"  Epoch: {best['Epoch']}, Val Loss: {val_loss_str}")

# Summary: Best fold overall (by Total_Facial_scale)
print("\n" + "="*80)
print("BEST FOLD OVERALL (by Total_Facial_scale R²)")
print("="*80)
total_df = results_df[results_df['Task'] == 'Total_Facial_scale']
if len(total_df) > 0:
    best_total = total_df.loc[total_df['R²'].idxmax()]
    
    # Format val_loss for display
    if pd.notna(best_total['Val_Loss']) and best_total['Val_Loss'] is not None:
        val_loss_str = f"{best_total['Val_Loss']:.4f}"
    else:
        val_loss_str = 'N/A'
    
    print(f"Fold {int(best_total['Fold'])}:")
    print(f"  Total_Facial_scale: R²={best_total['R²']:.3f}, r={best_total['r']:.3f}, MAE={best_total['MAE']:.3f}")
    print(f"  Epoch: {best_total['Epoch']}, Val Loss: {val_loss_str}")

# Comparison table: All folds side-by-side for key tasks
print("\n" + "="*80)
print("FOLD COMPARISON: KEY TASKS")
print("="*80)
key_tasks = ['Total_Facial_scale', 'Orbital_tightening', 'Nostril_muzzle', 'Tension_above_eyes']
for task in key_tasks:
    task_df = results_df[results_df['Task'] == task]
    if len(task_df) == 0:
        continue
    
    print(f"\n{task}:")
    print("Fold    R²        r        MAE      RMSE     Epoch")
    print("-" * 60)
    for _, row in task_df.iterrows():
        epoch_str = str(row['Epoch']) if pd.notna(row['Epoch']) else 'N/A'
        print(f"{int(row['Fold']):4d}  {row['R²']:8.3f}  {row['r']:8.3f}  {row['MAE']:7.3f}  {row['RMSE']:7.3f}  {epoch_str:>5s}")

# Moment-wise results
moment_df = pd.DataFrame(all_moment_results)
print("\n" + "="*80)
print("MOMENT-WISE METRICS (All Folds)")
print("="*80)

# Total_Facial_scale moment-wise comparison
total_moment = moment_df[(moment_df['Task'] == 'Total_Facial_scale')]
if len(total_moment) > 0:
    print("\nTotal_Facial_scale Moment-wise MAE by Fold:")
    print("Fold    M0      M1      M2      M3      M4")
    print("-" * 50)
    for fold in sorted(total_moment['Fold'].unique()):
        fold_data = total_moment[total_moment['Fold'] == fold]
        mae_values = []
        for moment in ['M0', 'M1', 'M2', 'M3', 'M4']:
            moment_data = fold_data[fold_data['Moment'] == moment]
            if len(moment_data) > 0:
                mae_values.append(f"{moment_data.iloc[0]['MAE']:.3f}")
            else:
                mae_values.append("N/A")
        print(f"{int(fold):4d}  {'  '.join(mae_values)}")

# Key tasks moment-wise summary
print("\n" + "="*80)
print("KEY TASKS MOMENT-WISE MAE (Best Fold)")
print("="*80)

# Find best fold (by Total_Facial_scale R²)
if len(total_df) > 0:
    best_fold = int(total_df.loc[total_df['R²'].idxmax(), 'Fold'])
    print(f"\nUsing Fold {best_fold} (best Total_Facial_scale R²):")
    
    for task in key_tasks:
        task_moment = moment_df[(moment_df['Task'] == task) & (moment_df['Fold'] == best_fold)]
        if len(task_moment) == 0:
            continue
        
        print(f"\n{task}:")
        print("Moment    M0      M1      M2      M3      M4")
        mae_by_moment = []
        for moment in ['M0', 'M1', 'M2', 'M3', 'M4']:
            moment_data = task_moment[task_moment['Moment'] == moment]
            if len(moment_data) > 0:
                mae_by_moment.append(f"{moment_data.iloc[0]['MAE']:.3f}")
            else:
                mae_by_moment.append("N/A")
        print(f"MAE     {'  '.join(mae_by_moment)}")

# Save results to CSV
output_dir = project_dir / 'evaluation_results'
output_dir.mkdir(exist_ok=True, parents=True)

results_csv = output_dir / f'evaluation_all_folds_{len(folds_to_evaluate)}folds.csv'
moment_csv = output_dir / f'evaluation_moment_wise_all_folds_{len(folds_to_evaluate)}folds.csv'

results_df.to_csv(results_csv, index=False)
moment_df.to_csv(moment_csv, index=False)

print("\n" + "="*80)
print("✅ EVALUATION COMPLETE!")
print("="*80)
print(f"\n📁 Results saved to:")
print(f"   - Overall: {results_csv}")
print(f"   - Moment-wise: {moment_csv}")

