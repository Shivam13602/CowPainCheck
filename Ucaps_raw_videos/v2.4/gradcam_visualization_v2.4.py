# ============================================================================
# GRAD-CAM VISUALIZATION - v2.4 (Regression Only)
# Visualizes what the model focuses on when making confident predictions
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
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import defaultdict
import warnings
import os
from typing import Dict, List, Tuple, Optional
warnings.filterwarnings('ignore')

# ============================================================================
# SETUP
# ============================================================================
print("="*80)
print("GRAD-CAM VISUALIZATION - v2.4 (Regression Only)")
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
output_dir = project_dir / 'gradcam_visualizations_v2.4'
output_dir.mkdir(exist_ok=True, parents=True)

splits_file = project_dir / 'train_val_test_splits_v2.json'
mapping_file = project_dir / 'sequence_label_mapping_v2.json'

print(f"\n📁 Paths:")
print(f"   Checkpoint dir: {checkpoint_dir} {'✅' if checkpoint_dir.exists() else '❌'}")
print(f"   Output dir: {output_dir} {'✅'}")
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
# MODEL ARCHITECTURE (Same as training, with hooks for Grad-CAM)
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
        
        # CNN - exact same structure as training script
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
            
            # Block 4 - Last conv layer (index 12 in Sequential)
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Global Average Pooling
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
        
        # Gradients and activations for Grad-CAM
        self.gradients = None
        self.activations = None
        
    def activations_hook(self, grad):
        self.gradients = grad
    
    def forward(self, x, return_activations=False):
        batch_size, num_frames, C, H, W = x.size()
        x = x.view(batch_size * num_frames, C, H, W)
        
        if return_activations:
            # Forward through CNN manually to capture activations
            # Block 1
            x = self.cnn[0](x)  # Conv
            x = self.cnn[1](x)  # BN
            x = self.cnn[2](x)  # ReLU
            x = self.cnn[3](x)  # MaxPool
            
            # Block 2
            x = self.cnn[4](x)  # Conv
            x = self.cnn[5](x)  # BN
            x = self.cnn[6](x)  # ReLU
            x = self.cnn[7](x)  # MaxPool
            
            # Block 3
            x = self.cnn[8](x)  # Conv
            x = self.cnn[9](x)  # BN
            x = self.cnn[10](x)  # ReLU
            x = self.cnn[11](x)  # MaxPool
            
            # Block 4 - Last conv layer (before pooling)
            x = self.cnn[12](x)  # Conv
            x = self.cnn[13](x)  # BN
            x = self.cnn[14](x)  # ReLU
            
            # Store activations and register hook
            self.activations = x
            if x.requires_grad:
                x.register_hook(self.activations_hook)
            
            # Continue forward
            x = self.cnn[15](x)  # MaxPool
            x = self.cnn[16](x)  # AdaptiveAvgPool
        else:
            # Normal forward pass
            x = self.cnn(x)
        
        cnn_features = x.view(batch_size * num_frames, -1)
        cnn_features = cnn_features.view(batch_size, num_frames, self.cnn_output_size)
        
        # LSTM
        lstm_out, _ = self.lstm(cnn_features)
        context, attention_weights = self.attention(lstm_out)
        context = self.dropout(context)
        
        # Outputs
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
    
    def clear_hooks(self):
        """Clear gradients and activations"""
        self.gradients = None
        self.activations = None

# ============================================================================
# GRAD-CAM IMPLEMENTATION
# ============================================================================
class GradCAM:
    def __init__(self, model, target_layer_name=None):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
    
    def generate_cam(self, input_tensor, target_task, target_index=0):
        """
        Generate Grad-CAM for a specific task
        
        Args:
            input_tensor: Input sequence (batch, frames, 3, H, W)
            target_task: Which feature to visualize (e.g., 'Orbital_tightening')
            target_index: Which sample in batch (default: 0)
        """
        self.model.train()  # Need gradients
        self.model.clear_hooks()
        
        # Forward pass with activations
        outputs, _ = self.model(input_tensor, return_activations=True)
        
        # Get target prediction
        if target_task not in outputs:
            raise ValueError(f"Task {target_task} not in model outputs")
        
        target = outputs[target_task][target_index]
        
        # Backward pass
        self.model.zero_grad()
        target.backward(retain_graph=True)
        
        # Get gradients and activations
        gradients = self.model.gradients  # (batch*frames, 256, H', W')
        activations = self.model.activations  # (batch*frames, 256, H', W')
        
        if gradients is None or activations is None:
            raise ValueError("Gradients or activations not captured. Check hook registration.")
        
        batch_size, num_frames = input_tensor.size(0), input_tensor.size(1)
        
        # Process each frame
        cams = []
        for frame_idx in range(num_frames):
            idx = target_index * num_frames + frame_idx
            if idx >= gradients.size(0):
                # Handle case where we have fewer frames than expected
                idx = min(idx, gradients.size(0) - 1)
            
            frame_grads = gradients[idx]  # (256, H', W')
            frame_acts = activations[idx]  # (256, H', W')
            
            # Global average pooling of gradients
            weights = torch.mean(frame_grads, dim=(1, 2), keepdim=True)  # (256, 1, 1)
            
            # Weighted combination of activation maps
            cam = torch.sum(weights * frame_acts, dim=0)  # (H', W')
            
            # ReLU
            cam = F.relu(cam)
            
            # Normalize
            cam = cam - cam.min()
            if cam.max() > 0:
                cam = cam / cam.max()
            
            cams.append(cam.detach().cpu().numpy())
        
        self.model.eval()  # Back to eval mode
        return cams

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================
def load_sequence_frames(sequence_path, max_frames=32):
    """Load frames from sequence directory"""
    frame_files = sorted(list(sequence_path.glob('*.jpg')) + list(sequence_path.glob('*.png')))
    
    if len(frame_files) == 0:
        return None, None
    
    if len(frame_files) > max_frames:
        indices = np.linspace(0, len(frame_files)-1, max_frames, dtype=int)
        frame_files = [frame_files[i] for i in indices]
    
    frames = []
    original_frames = []
    
    for frame_file in frame_files:
        img = Image.open(frame_file).convert('RGB')
        original_frames.append(np.array(img))
        
        # Resize for model
        img_resized = img.resize((112, 112))
        frames.append(np.array(img_resized))
    
    return frames, original_frames

def apply_colormap_on_image(img, activation, colormap_name='jet'):
    """Apply colormap on image"""
    heatmap = cm.get_cmap(colormap_name)(activation)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)
    
    # Resize heatmap to match image
    if heatmap.shape[:2] != img.shape[:2]:
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Blend
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    return overlay, heatmap

def visualize_gradcam_sequence(
    model,
    sequence_path,
    sequence_info,
    output_dir,
    fold_idx,
    target_features=['Orbital_tightening', 'Total_Facial_scale'],
    num_frames_to_show=8
):
    """Generate and save Grad-CAM visualizations for a sequence"""
    
    # Load frames
    frames, original_frames = load_sequence_frames(sequence_path, max_frames=32)
    if frames is None:
        print(f"   ⚠️  No frames found in {sequence_path}")
        return
    
    # Prepare input tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    frame_tensors = []
    for frame in frames:
        frame_pil = Image.fromarray(frame)
        frame_tensor = transform(frame_pil)
        frame_tensors.append(frame_tensor)
    
    input_tensor = torch.stack(frame_tensors).unsqueeze(0)  # (1, frames, 3, H, W)
    input_tensor = input_tensor.to(next(model.parameters()).device)
    input_tensor.requires_grad = True
    
    # Initialize Grad-CAM
    gradcam = GradCAM(model)
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        outputs, attention_weights = model(input_tensor, return_activations=False)
    
    # Generate CAMs for each target feature
    all_cams = {}
    predictions = {}
    
    for feature in target_features:
        if feature not in outputs:
            continue
        
        try:
            cams = gradcam.generate_cam(input_tensor, feature, target_index=0)
            all_cams[feature] = cams
            predictions[feature] = outputs[feature][0].item()
        except Exception as e:
            print(f"   ⚠️  Error generating CAM for {feature}: {e}")
            continue
    
    # Select frames to visualize (evenly spaced)
    num_frames = len(original_frames)
    frame_indices = np.linspace(0, num_frames-1, min(num_frames_to_show, num_frames), dtype=int)
    
    # Create visualization
    animal_id = sequence_info.get('animal', sequence_info.get('animal_id', 'unknown'))
    moment = sequence_info.get('moment', 'unknown')
    seq_id = sequence_info.get('sequence_id', 'unknown')
    
    save_dir = output_dir / f'fold_{fold_idx}' / f'animal_{animal_id}_moment_{moment}'
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # Create grid visualization
    num_features = len(all_cams)
    fig, axes = plt.subplots(num_features, len(frame_indices), 
                            figsize=(3*len(frame_indices), 3*num_features))
    
    if num_features == 1:
        axes = axes.reshape(1, -1)
    
    for feat_idx, (feature, cams) in enumerate(all_cams.items()):
        pred_value = predictions.get(feature, 0.0)
        
        for col_idx, frame_idx in enumerate(frame_indices):
            ax = axes[feat_idx, col_idx]
            
            # Get original frame and CAM
            original_frame = original_frames[frame_idx]
            cam = cams[frame_idx]
            
            # Upsample CAM to original frame size
            cam_upsampled = cv2.resize(cam, (original_frame.shape[1], original_frame.shape[0]))
            
            # Apply colormap
            overlay, heatmap = apply_colormap_on_image(original_frame, cam_upsampled)
            
            # Display
            ax.imshow(overlay)
            ax.set_title(f'{feature}\nPred: {pred_value:.2f}\nFrame {frame_idx+1}', 
                        fontsize=8)
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_dir / f'gradcam_grid_{seq_id}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save individual frame visualizations for best feature
    if 'Total_Facial_scale' in all_cams:
        best_feature = 'Total_Facial_scale'
    elif len(all_cams) > 0:
        best_feature = list(all_cams.keys())[0]
    else:
        return
    
    for frame_idx in frame_indices:
        original_frame = original_frames[frame_idx]
        cam = all_cams[best_feature][frame_idx]
        cam_upsampled = cv2.resize(cam, (original_frame.shape[1], original_frame.shape[0]))
        overlay, _ = apply_colormap_on_image(original_frame, cam_upsampled)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(original_frame)
        axes[0].set_title('Original Frame')
        axes[0].axis('off')
        
        axes[1].imshow(cam_upsampled, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].axis('off')
        
        axes[2].imshow(overlay)
        axes[2].set_title(f'Overlay\n{best_feature}: {predictions[best_feature]:.2f}')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_dir / f'frame_{frame_idx:03d}_{best_feature}.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"   ✅ Saved visualizations to {save_dir}")

# ============================================================================
# MAIN VISUALIZATION
# ============================================================================
print("\n" + "="*80)
print("GENERATING GRAD-CAM VISUALIZATIONS")
print("="*80)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n✅ Device: {device}")

# Select sequences to visualize (high confidence predictions)
# Focus on test set or validation sequences with high predictions
target_features = ['Orbital_tightening', 'Total_Facial_scale', 'Ears_lateral']

# Get test sequences
test_animals = splits.get('test_animals', [14, 17])
test_sequences = [s for s in all_sequences if s.get('animal', s.get('animal_id')) in test_animals]

print(f"\n📊 Found {len(test_sequences)} test sequences")
print(f"   Will visualize sequences with high confidence predictions")

# Select a few sequences to visualize
sequences_to_visualize = test_sequences[:5]  # First 5 test sequences

# Load best model from each fold and visualize
for fold_idx in range(9):
    print(f"\n📊 Processing Fold {fold_idx}...")
    
    best_model_path = checkpoint_dir / f'best_model_v2.4_fold_{fold_idx}.pt'
    if not best_model_path.exists():
        print(f"   ⚠️  Model not found: {best_model_path.name}")
        continue
    
    # Load model
    checkpoint = torch.load(best_model_path, map_location=device)
    model = TemporalPainModel_v2_4(num_frames=32, lstm_hidden_size=128, use_bidirectional=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Visualize selected sequences
    for seq_info in sequences_to_visualize:
        # Find sequence path
        seq_id = seq_info.get('sequence_id', '')
        if 'sequence_path' in seq_info:
            seq_path = sequence_dir / seq_info['sequence_path']
        elif seq_id:
            seq_path = sequence_dir / seq_id
        else:
            continue
        
        if not seq_path.exists():
            # Try to find it
            possible_paths = [
                sequence_dir / seq_id,
                sequence_dir / f'sequence_{seq_id}',
            ]
            found = False
            for p in possible_paths:
                if p.exists():
                    seq_path = p
                    found = True
                    break
            if not found:
                continue
        
        print(f"   📹 Visualizing {seq_id}...")
        try:
            visualize_gradcam_sequence(
                model, seq_path, seq_info, output_dir, fold_idx,
                target_features=target_features,
                num_frames_to_show=8
            )
        except Exception as e:
            print(f"   ⚠️  Error visualizing {seq_id}: {e}")
            continue

print("\n" + "="*80)
print("✅ GRAD-CAM VISUALIZATION COMPLETE")
print("="*80)
print(f"\n📁 Visualizations saved to: {output_dir}")
print("\n💡 Next: Review visualizations to understand model attention patterns")

