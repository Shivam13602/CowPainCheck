# ============================================================================
# PUBLICATION-QUALITY VISUALIZATIONS FOR MODEL EVALUATION RESULTS
# Generates journal-ready figures for peer-reviewed paper
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
warnings.filterwarnings('ignore')

# Set publication-quality style
try:
    plt.style.use('seaborn-v0_8-paper')
except:
    try:
        plt.style.use('seaborn-paper')
    except:
        plt.style.use('default')
sns.set_palette("husl")
sns.set_context("paper", font_scale=1.2)

# Publication settings
PUBLICATION_FIGSIZE = (10, 6)
DPI = 300  # High resolution for publications
FONT_SIZE = 12
TITLE_SIZE = 14
LABEL_SIZE = 11

# Color scheme for moments
MOMENT_COLORS = {
    'M0': '#2ecc71',  # Green (baseline, no pain)
    'M1': '#3498db',  # Blue (mild pain)
    'M2': '#e74c3c',  # Red (acute/severe pain)
    'M3': '#f39c12',  # Orange (moderate pain)
    'M4': '#95a5a6'   # Gray (recovering)
}

# Feature display names (for publication)
FEATURE_DISPLAY_NAMES = {
    'Orbital_tightening': 'Orbital Tightening',
    'Tension_above_eyes': 'Tension Above Eyes',
    'Cheek_tightening': 'Cheek Tightening',
    'Ears_frontal': 'Ears Frontal',
    'Ears_lateral': 'Ears Lateral',
    'Lip_jaw_profile': 'Lip/Jaw Profile',
    'Nostril_muzzle': 'Nostril/Muzzle',
    'Total_Facial_scale': 'Total Facial Scale'
}

def setup_paths():
    """Setup paths for data files"""
    # Try local paths first
    base_path = Path('.')
    
    # Check for evaluation results
    possible_paths = [
        base_path / 'evaluation_results',
        base_path / 'facial_pain_project_v2' / 'evaluation_results',
        Path('/content/drive/MyDrive/facial_pain_project_v2/evaluation_results'),
        Path('/content/drive/MyDrive/evaluation_results'),
    ]
    
    results_dir = None
    for path in possible_paths:
        if path.exists():
            results_dir = path
            break
    
    if results_dir is None:
        print("⚠️  Evaluation results directory not found. Will try to find CSV files...")
        # Try to find CSV files directly
        csv_files = list(base_path.rglob('evaluation_all_folds_*.csv'))
        if csv_files:
            results_dir = csv_files[0].parent
            print(f"✅ Found results in: {results_dir}")
        else:
            raise FileNotFoundError("Could not find evaluation results. Please run evaluate_model_v2.py first.")
    
    return results_dir

def load_evaluation_data(results_dir):
    """Load evaluation CSV files"""
    # Find CSV files
    overall_csv = None
    moment_csv = None
    
    for csv_file in results_dir.glob('evaluation_all_folds_*.csv'):
        overall_csv = csv_file
        break
    
    for csv_file in results_dir.glob('evaluation_moment_wise_*.csv'):
        moment_csv = csv_file
        break
    
    if overall_csv is None:
        raise FileNotFoundError(f"Could not find evaluation_all_folds_*.csv in {results_dir}")
    
    print(f"✅ Loading: {overall_csv.name}")
    results_df = pd.read_csv(overall_csv)
    
    if moment_csv:
        print(f"✅ Loading: {moment_csv.name}")
        moment_df = pd.read_csv(moment_csv)
    else:
        print("⚠️  Moment-wise CSV not found, will skip moment-wise visualizations")
        moment_df = None
    
    return results_df, moment_df

def plot_mae_per_fold(results_df, output_dir):
    """Figure 1: MAE per fold for all tasks"""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    tasks = ['Total_Facial_scale', 'Orbital_tightening', 'Tension_above_eyes', 
             'Cheek_tightening', 'Ears_frontal', 'Ears_lateral', 
             'Lip_jaw_profile', 'Nostril_muzzle']
    
    for idx, task in enumerate(tasks):
        ax = axes[idx]
        task_data = results_df[results_df['Task'] == task].copy()
        
        if len(task_data) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(FEATURE_DISPLAY_NAMES.get(task, task))
            continue
        
        # Sort by fold
        task_data = task_data.sort_values('Fold')
        
        # Create bar plot
        bars = ax.bar(task_data['Fold'], task_data['MAE'], 
                     color=sns.color_palette("husl", len(task_data)),
                     edgecolor='black', linewidth=1.5, alpha=0.8)
        
        # Highlight best fold (lowest MAE)
        best_idx = task_data['MAE'].idxmin()
        bars[task_data.index.get_loc(best_idx)].set_color('#e74c3c')
        bars[task_data.index.get_loc(best_idx)].set_edgecolor('black')
        bars[task_data.index.get_loc(best_idx)].set_linewidth(2.5)
        
        ax.set_xlabel('Fold', fontsize=LABEL_SIZE, fontweight='bold')
        ax.set_ylabel('MAE', fontsize=LABEL_SIZE, fontweight='bold')
        ax.set_title(FEATURE_DISPLAY_NAMES.get(task, task), 
                    fontsize=TITLE_SIZE, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_xticks(task_data['Fold'])
        
        # Add value labels on bars
        for i, (fold, mae) in enumerate(zip(task_data['Fold'], task_data['MAE'])):
            ax.text(fold, mae + mae*0.02, f'{mae:.3f}', 
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.suptitle('Mean Absolute Error (MAE) per Fold Across All Tasks', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    output_path = output_dir / 'figure_1_mae_per_fold.png'
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_path.name}")
    plt.close()

def plot_r2_per_fold(results_df, output_dir):
    """Figure 2: R² per fold for all tasks"""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    tasks = ['Total_Facial_scale', 'Orbital_tightening', 'Tension_above_eyes', 
             'Cheek_tightening', 'Ears_frontal', 'Ears_lateral', 
             'Lip_jaw_profile', 'Nostril_muzzle']
    
    for idx, task in enumerate(tasks):
        ax = axes[idx]
        task_data = results_df[results_df['Task'] == task].copy()
        
        if len(task_data) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(FEATURE_DISPLAY_NAMES.get(task, task))
            continue
        
        task_data = task_data.sort_values('Fold')
        
        # Create bar plot
        colors = ['#e74c3c' if r2 < 0 else '#2ecc71' for r2 in task_data['R²']]
        bars = ax.bar(task_data['Fold'], task_data['R²'], 
                     color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
        
        # Highlight best fold (highest R²)
        best_idx = task_data['R²'].idxmax()
        bars[task_data.index.get_loc(best_idx)].set_color('#3498db')
        bars[task_data.index.get_loc(best_idx)].set_edgecolor('black')
        bars[task_data.index.get_loc(best_idx)].set_linewidth(2.5)
        
        # Add zero line
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        ax.set_xlabel('Fold', fontsize=LABEL_SIZE, fontweight='bold')
        ax.set_ylabel('R²', fontsize=LABEL_SIZE, fontweight='bold')
        ax.set_title(FEATURE_DISPLAY_NAMES.get(task, task), 
                    fontsize=TITLE_SIZE, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_xticks(task_data['Fold'])
        
        # Add value labels
        for i, (fold, r2) in enumerate(zip(task_data['Fold'], task_data['R²'])):
            ax.text(fold, r2 + (0.05 if r2 >= 0 else -0.05), f'{r2:.3f}', 
                   ha='center', va='bottom' if r2 >= 0 else 'top', 
                   fontsize=9, fontweight='bold')
    
    plt.suptitle('Coefficient of Determination (R²) per Fold Across All Tasks', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    output_path = output_dir / 'figure_2_r2_per_fold.png'
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_path.name}")
    plt.close()

def plot_correlation_per_fold(results_df, output_dir):
    """Figure 3: Pearson correlation (r) per fold"""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    tasks = ['Total_Facial_scale', 'Orbital_tightening', 'Tension_above_eyes', 
             'Cheek_tightening', 'Ears_frontal', 'Ears_lateral', 
             'Lip_jaw_profile', 'Nostril_muzzle']
    
    for idx, task in enumerate(tasks):
        ax = axes[idx]
        task_data = results_df[results_df['Task'] == task].copy()
        
        if len(task_data) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(FEATURE_DISPLAY_NAMES.get(task, task))
            continue
        
        task_data = task_data.sort_values('Fold')
        
        # Create bar plot
        colors = ['#e74c3c' if r < 0 else '#2ecc71' for r in task_data['r']]
        bars = ax.bar(task_data['Fold'], task_data['r'], 
                     color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
        
        # Highlight best fold
        best_idx = task_data['r'].idxmax()
        bars[task_data.index.get_loc(best_idx)].set_color('#3498db')
        bars[task_data.index.get_loc(best_idx)].set_edgecolor('black')
        bars[task_data.index.get_loc(best_idx)].set_linewidth(2.5)
        
        # Add zero line
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        ax.set_xlabel('Fold', fontsize=LABEL_SIZE, fontweight='bold')
        ax.set_ylabel('Pearson r', fontsize=LABEL_SIZE, fontweight='bold')
        ax.set_title(FEATURE_DISPLAY_NAMES.get(task, task), 
                    fontsize=TITLE_SIZE, fontweight='bold')
        ax.set_ylim([-0.5, 0.8])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_xticks(task_data['Fold'])
        
        # Add value labels
        for i, (fold, r) in enumerate(zip(task_data['Fold'], task_data['r'])):
            ax.text(fold, r + (0.03 if r >= 0 else -0.03), f'{r:.3f}', 
                   ha='center', va='bottom' if r >= 0 else 'top', 
                   fontsize=9, fontweight='bold')
    
    plt.suptitle('Pearson Correlation Coefficient (r) per Fold Across All Tasks', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    output_path = output_dir / 'figure_3_correlation_per_fold.png'
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_path.name}")
    plt.close()

def plot_moment_wise_mae(moment_df, output_dir):
    """Figure 4: Moment-wise MAE heatmap"""
    if moment_df is None:
        print("⚠️  Skipping moment-wise visualizations (no data)")
        return
    
    # Focus on Total_Facial_scale and key features
    key_tasks = ['Total_Facial_scale', 'Orbital_tightening', 'Ears_frontal', 'Nostril_muzzle']
    
    fig, axes = plt.subplots(1, len(key_tasks), figsize=(16, 4))
    if len(key_tasks) == 1:
        axes = [axes]
    
    for idx, task in enumerate(key_tasks):
        ax = axes[idx]
        task_data = moment_df[moment_df['Task'] == task].copy()
        
        if len(task_data) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            continue
        
        # Create pivot table: Fold × Moment
        pivot_data = task_data.pivot_table(
            values='MAE', 
            index='Fold', 
            columns='Moment',
            aggfunc='mean'
        )
        
        # Reorder moments
        moment_order = ['M0', 'M1', 'M2', 'M3', 'M4']
        pivot_data = pivot_data.reindex(columns=[m for m in moment_order if m in pivot_data.columns])
        pivot_data = pivot_data.sort_index()
        
        # Create heatmap
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlOrRd', 
                   cbar_kws={'label': 'MAE'}, ax=ax, 
                   linewidths=1, linecolor='black', square=False,
                   annot_kws={'fontsize': 10, 'fontweight': 'bold'})
        
        ax.set_title(FEATURE_DISPLAY_NAMES.get(task, task), 
                    fontsize=TITLE_SIZE, fontweight='bold', pad=15)
        ax.set_xlabel('Moment', fontsize=LABEL_SIZE, fontweight='bold')
        ax.set_ylabel('Fold', fontsize=LABEL_SIZE, fontweight='bold')
    
    plt.suptitle('Moment-wise Mean Absolute Error (MAE) Heatmap', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_path = output_dir / 'figure_4_moment_wise_mae_heatmap.png'
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_path.name}")
    plt.close()

def plot_moment_wise_comparison(moment_df, output_dir):
    """Figure 5: Moment-wise MAE comparison (bar chart)"""
    if moment_df is None:
        print("⚠️  Skipping moment-wise comparison (no data)")
        return
    
    # Focus on Total_Facial_scale
    total_data = moment_df[moment_df['Task'] == 'Total_Facial_scale'].copy()
    
    if len(total_data) == 0:
        print("⚠️  No Total_Facial_scale data for moment comparison")
        return
    
    # Group by moment and calculate mean MAE across all folds
    moment_stats = total_data.groupby('Moment')['MAE'].agg(['mean', 'std', 'count']).reset_index()
    moment_stats = moment_stats.sort_values('Moment', key=lambda x: x.str[1].astype(int))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bar plot with error bars
    moments = moment_stats['Moment']
    means = moment_stats['mean']
    stds = moment_stats['std']
    
    colors = [MOMENT_COLORS.get(m, '#95a5a6') for m in moments]
    bars = ax.bar(moments, means, yerr=stds, color=colors, 
                 edgecolor='black', linewidth=2, alpha=0.8, 
                 capsize=10, capthick=2, error_kw={'linewidth': 2})
    
    # Add value labels
    for i, (moment, mean, std) in enumerate(zip(moments, means, stds)):
        ax.text(moment, mean + std + mean*0.05, f'{mean:.3f}±{std:.3f}', 
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Moment', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_ylabel('Mean Absolute Error (MAE)', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_title('Moment-wise Performance: Total Facial Scale MAE Across All Folds', 
                fontsize=TITLE_SIZE, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add moment descriptions
    moment_labels = {
        'M0': 'Baseline\n(Pre-surgery)',
        'M1': 'Early Post-op\n(~30 min)',
        'M2': 'Peak Pain\n(~2-4 hours)',
        'M3': 'Declining\n(~6-8 hours)',
        'M4': 'Residual\n(~24 hours)'
    }
    
    # Add text annotations
    for i, moment in enumerate(moments):
        if moment in moment_labels:
            ax.text(i, -0.3, moment_labels[moment], ha='center', va='top', 
                   fontsize=9, style='italic')
    
    plt.tight_layout()
    
    output_path = output_dir / 'figure_5_moment_wise_comparison.png'
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_path.name}")
    plt.close()

def plot_fold_comparison_summary(results_df, output_dir):
    """Figure 6: Summary comparison of all folds (key metrics)"""
    # Focus on Total_Facial_scale
    total_data = results_df[results_df['Task'] == 'Total_Facial_scale'].copy()
    total_data = total_data.sort_values('Fold')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Subplot 1: MAE
    ax1 = axes[0, 0]
    bars1 = ax1.bar(total_data['Fold'], total_data['MAE'], 
                   color=sns.color_palette("husl", len(total_data)),
                   edgecolor='black', linewidth=2, alpha=0.8)
    best_idx = total_data['MAE'].idxmin()
    bars1[total_data.index.get_loc(best_idx)].set_color('#e74c3c')
    bars1[total_data.index.get_loc(best_idx)].set_linewidth(3)
    ax1.set_xlabel('Fold', fontsize=LABEL_SIZE, fontweight='bold')
    ax1.set_ylabel('MAE', fontsize=LABEL_SIZE, fontweight='bold')
    ax1.set_title('Mean Absolute Error (MAE)', fontsize=TITLE_SIZE, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    for i, (fold, mae) in enumerate(zip(total_data['Fold'], total_data['MAE'])):
        ax1.text(fold, mae + mae*0.02, f'{mae:.3f}', ha='center', va='bottom', 
               fontsize=10, fontweight='bold')
    
    # Subplot 2: R²
    ax2 = axes[0, 1]
    colors2 = ['#e74c3c' if r2 < 0 else '#2ecc71' for r2 in total_data['R²']]
    bars2 = ax2.bar(total_data['Fold'], total_data['R²'], 
                   color=colors2, edgecolor='black', linewidth=2, alpha=0.8)
    best_idx2 = total_data['R²'].idxmax()
    bars2[total_data.index.get_loc(best_idx2)].set_color('#3498db')
    bars2[total_data.index.get_loc(best_idx2)].set_linewidth(3)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Fold', fontsize=LABEL_SIZE, fontweight='bold')
    ax2.set_ylabel('R²', fontsize=LABEL_SIZE, fontweight='bold')
    ax2.set_title('Coefficient of Determination (R²)', fontsize=TITLE_SIZE, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    for i, (fold, r2) in enumerate(zip(total_data['Fold'], total_data['R²'])):
        ax2.text(fold, r2 + (0.02 if r2 >= 0 else -0.02), f'{r2:.3f}', 
               ha='center', va='bottom' if r2 >= 0 else 'top', 
               fontsize=10, fontweight='bold')
    
    # Subplot 3: Correlation
    ax3 = axes[1, 0]
    colors3 = ['#e74c3c' if r < 0 else '#2ecc71' for r in total_data['r']]
    bars3 = ax3.bar(total_data['Fold'], total_data['r'], 
                   color=colors3, edgecolor='black', linewidth=2, alpha=0.8)
    best_idx3 = total_data['r'].idxmax()
    bars3[total_data.index.get_loc(best_idx3)].set_color('#3498db')
    bars3[total_data.index.get_loc(best_idx3)].set_linewidth(3)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax3.set_xlabel('Fold', fontsize=LABEL_SIZE, fontweight='bold')
    ax3.set_ylabel('Pearson r', fontsize=LABEL_SIZE, fontweight='bold')
    ax3.set_title('Pearson Correlation Coefficient (r)', fontsize=TITLE_SIZE, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    for i, (fold, r) in enumerate(zip(total_data['Fold'], total_data['r'])):
        ax3.text(fold, r + (0.02 if r >= 0 else -0.02), f'{r:.3f}', 
               ha='center', va='bottom' if r >= 0 else 'top', 
               fontsize=10, fontweight='bold')
    
    # Subplot 4: RMSE
    ax4 = axes[1, 1]
    bars4 = ax4.bar(total_data['Fold'], total_data['RMSE'], 
                   color=sns.color_palette("husl", len(total_data)),
                   edgecolor='black', linewidth=2, alpha=0.8)
    best_idx4 = total_data['RMSE'].idxmin()
    bars4[total_data.index.get_loc(best_idx4)].set_color('#e74c3c')
    bars4[total_data.index.get_loc(best_idx4)].set_linewidth(3)
    ax4.set_xlabel('Fold', fontsize=LABEL_SIZE, fontweight='bold')
    ax4.set_ylabel('RMSE', fontsize=LABEL_SIZE, fontweight='bold')
    ax4.set_title('Root Mean Squared Error (RMSE)', fontsize=TITLE_SIZE, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    for i, (fold, rmse) in enumerate(zip(total_data['Fold'], total_data['RMSE'])):
        ax4.text(fold, rmse + rmse*0.02, f'{rmse:.3f}', ha='center', va='bottom', 
               fontsize=10, fontweight='bold')
    
    plt.suptitle('Model Performance Summary: Total Facial Scale Across All Folds', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    output_path = output_dir / 'figure_6_fold_comparison_summary.png'
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_path.name}")
    plt.close()

def plot_task_performance_comparison(results_df, output_dir):
    """Figure 7: Performance comparison across all tasks (best fold)"""
    # Get best fold for each task (by R²)
    best_per_task = []
    for task in results_df['Task'].unique():
        task_data = results_df[results_df['Task'] == task].copy()
        if len(task_data) > 0:
            best = task_data.loc[task_data['R²'].idxmax()]
            best_per_task.append(best)
    
    best_df = pd.DataFrame(best_per_task)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Sort by MAE for better visualization
    best_df = best_df.sort_values('MAE')
    
    # Subplot 1: MAE
    ax1 = axes[0, 0]
    colors1 = ['#e74c3c' if task == 'Total_Facial_scale' else '#3498db' 
              for task in best_df['Task']]
    bars1 = ax1.barh(range(len(best_df)), best_df['MAE'], 
                    color=colors1, edgecolor='black', linewidth=1.5, alpha=0.8)
    ax1.set_yticks(range(len(best_df)))
    ax1.set_yticklabels([FEATURE_DISPLAY_NAMES.get(t, t) for t in best_df['Task']], 
                        fontsize=10)
    ax1.set_xlabel('MAE', fontsize=LABEL_SIZE, fontweight='bold')
    ax1.set_title('Mean Absolute Error (MAE) - Best Fold per Task', 
                 fontsize=TITLE_SIZE, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    for i, mae in enumerate(best_df['MAE']):
        ax1.text(mae + mae*0.02, i, f'{mae:.3f}', ha='left', va='center', 
               fontsize=9, fontweight='bold')
    
    # Subplot 2: R²
    ax2 = axes[0, 1]
    colors2 = ['#e74c3c' if task == 'Total_Facial_scale' else 
              ('#2ecc71' if r2 >= 0 else '#e74c3c') 
              for task, r2 in zip(best_df['Task'], best_df['R²'])]
    bars2 = ax2.barh(range(len(best_df)), best_df['R²'], 
                    color=colors2, edgecolor='black', linewidth=1.5, alpha=0.8)
    ax2.set_yticks(range(len(best_df)))
    ax2.set_yticklabels([FEATURE_DISPLAY_NAMES.get(t, t) for t in best_df['Task']], 
                        fontsize=10)
    ax2.set_xlabel('R²', fontsize=LABEL_SIZE, fontweight='bold')
    ax2.set_title('Coefficient of Determination (R²) - Best Fold per Task', 
                 fontsize=TITLE_SIZE, fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    for i, r2 in enumerate(best_df['R²']):
        ax2.text(r2 + (0.01 if r2 >= 0 else -0.01), i, f'{r2:.3f}', 
               ha='left' if r2 >= 0 else 'right', va='center', 
               fontsize=9, fontweight='bold')
    
    # Subplot 3: Correlation
    ax3 = axes[1, 0]
    colors3 = ['#e74c3c' if task == 'Total_Facial_scale' else 
              ('#2ecc71' if r >= 0 else '#e74c3c') 
              for task, r in zip(best_df['Task'], best_df['r'])]
    bars3 = ax3.barh(range(len(best_df)), best_df['r'], 
                    color=colors3, edgecolor='black', linewidth=1.5, alpha=0.8)
    ax3.set_yticks(range(len(best_df)))
    ax3.set_yticklabels([FEATURE_DISPLAY_NAMES.get(t, t) for t in best_df['Task']], 
                        fontsize=10)
    ax3.set_xlabel('Pearson r', fontsize=LABEL_SIZE, fontweight='bold')
    ax3.set_title('Pearson Correlation (r) - Best Fold per Task', 
                 fontsize=TITLE_SIZE, fontweight='bold')
    ax3.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax3.grid(axis='x', alpha=0.3, linestyle='--')
    for i, r in enumerate(best_df['r']):
        ax3.text(r + (0.02 if r >= 0 else -0.02), i, f'{r:.3f}', 
               ha='left' if r >= 0 else 'right', va='center', 
               fontsize=9, fontweight='bold')
    
    # Subplot 4: RMSE
    ax4 = axes[1, 1]
    colors4 = ['#e74c3c' if task == 'Total_Facial_scale' else '#3498db' 
              for task in best_df['Task']]
    bars4 = ax4.barh(range(len(best_df)), best_df['RMSE'], 
                    color=colors4, edgecolor='black', linewidth=1.5, alpha=0.8)
    ax4.set_yticks(range(len(best_df)))
    ax4.set_yticklabels([FEATURE_DISPLAY_NAMES.get(t, t) for t in best_df['Task']], 
                        fontsize=10)
    ax4.set_xlabel('RMSE', fontsize=LABEL_SIZE, fontweight='bold')
    ax4.set_title('Root Mean Squared Error (RMSE) - Best Fold per Task', 
                 fontsize=TITLE_SIZE, fontweight='bold')
    ax4.grid(axis='x', alpha=0.3, linestyle='--')
    for i, rmse in enumerate(best_df['RMSE']):
        ax4.text(rmse + rmse*0.02, i, f'{rmse:.3f}', ha='left', va='center', 
               fontsize=9, fontweight='bold')
    
    plt.suptitle('Best Model Performance per Task (Best Fold Selected)', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    output_path = output_dir / 'figure_7_task_performance_comparison.png'
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_path.name}")
    plt.close()

def plot_predictions_vs_ground_truth(results_df, output_dir, test_predictions_file=None):
    """Figure 8: Scatter plots of predictions vs ground truth (if available)"""
    # Try to load predictions if available
    if test_predictions_file and Path(test_predictions_file).exists():
        try:
            pred_df = pd.read_csv(test_predictions_file)
            print(f"✅ Loaded predictions from: {test_predictions_file}")
        except:
            print("⚠️  Could not load predictions file, skipping scatter plots")
            return
    else:
        # Try to find predictions file
        possible_files = [
            Path('test_set_predictions_v2_fold3.csv'),
            Path('../test_set_predictions_v2_fold3.csv'),
        ]
        pred_df = None
        for f in possible_files:
            if f.exists():
                try:
                    pred_df = pd.read_csv(f)
                    print(f"✅ Found predictions file: {f}")
                    break
                except:
                    continue
        
        if pred_df is None:
            print("⚠️  No predictions file found. Scatter plots require test_set_predictions CSV.")
            print("   To generate: Run evaluation script and save individual predictions.")
            return
    
    # Key tasks for scatter plots
    key_tasks = ['Total_Facial_scale', 'Orbital_tightening', 'Ears_frontal', 'Nostril_muzzle']
    
    # Find best fold for Total_Facial_scale
    total_data = results_df[results_df['Task'] == 'Total_Facial_scale']
    if len(total_data) > 0:
        best_fold = int(total_data.loc[total_data['R²'].idxmax(), 'Fold'])
    else:
        best_fold = 3  # Default
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, task in enumerate(key_tasks):
        ax = axes[idx]
        
        # Get predictions and targets for this task
        pred_col = f'{task}_pred' if f'{task}_pred' in pred_df.columns else None
        target_col = task if task in pred_df.columns else f'{task}_target'
        
        if pred_col is None or target_col not in pred_df.columns:
            ax.text(0.5, 0.5, 'Data not available', ha='center', va='center')
            ax.set_title(FEATURE_DISPLAY_NAMES.get(task, task))
            continue
        
        predictions = pred_df[pred_col].values
        targets = pred_df[target_col].values
        
        # Remove NaN values
        mask = ~(np.isnan(predictions) | np.isnan(targets))
        predictions = predictions[mask]
        targets = targets[mask]
        
        if len(predictions) == 0:
            ax.text(0.5, 0.5, 'No valid data', ha='center', va='center')
            continue
        
        # Create scatter plot
        ax.scatter(targets, predictions, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        
        # Add perfect prediction line (y=x)
        min_val = min(min(targets), min(predictions))
        max_val = max(max(targets), max(predictions))
        ax.plot([min_val, max_val], [min_val, max_val], 
               'r--', linewidth=2, label='Perfect Prediction', alpha=0.8)
        
        # Calculate metrics
        from scipy.stats import pearsonr
        from sklearn.metrics import r2_score, mean_absolute_error
        
        r, p = pearsonr(targets, predictions)
        r2 = r2_score(targets, predictions)
        mae = mean_absolute_error(targets, predictions)
        
        # Add metrics text
        metrics_text = f'R² = {r2:.3f}\nr = {r:.3f}\nMAE = {mae:.3f}'
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               fontweight='bold')
        
        ax.set_xlabel('Ground Truth', fontsize=LABEL_SIZE, fontweight='bold')
        ax.set_ylabel('Predicted', fontsize=LABEL_SIZE, fontweight='bold')
        ax.set_title(FEATURE_DISPLAY_NAMES.get(task, task), 
                    fontsize=TITLE_SIZE, fontweight='bold')
        ax.grid(alpha=0.3, linestyle='--')
        ax.legend(loc='lower right', fontsize=10)
    
    plt.suptitle(f'Predictions vs Ground Truth (Best Fold: {best_fold})', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    output_path = output_dir / 'figure_8_predictions_vs_ground_truth.png'
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_path.name}")
    plt.close()

def create_summary_table(results_df, moment_df, output_dir):
    """Create summary statistics table"""
    # Overall summary
    summary_data = []
    
    for task in results_df['Task'].unique():
        task_data = results_df[results_df['Task'] == task].copy()
        if len(task_data) == 0:
            continue
        
        summary_data.append({
            'Task': FEATURE_DISPLAY_NAMES.get(task, task),
            'Best_Fold': int(task_data.loc[task_data['R²'].idxmax(), 'Fold']),
            'Best_R²': task_data['R²'].max(),
            'Best_r': task_data.loc[task_data['R²'].idxmax(), 'r'],
            'Best_MAE': task_data.loc[task_data['R²'].idxmax(), 'MAE'],
            'Best_RMSE': task_data.loc[task_data['R²'].idxmax(), 'RMSE'],
            'Mean_R²': task_data['R²'].mean(),
            'Mean_MAE': task_data['MAE'].mean(),
            'Std_MAE': task_data['MAE'].std(),
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Best_R²', ascending=False)
    
    # Save to CSV
    summary_csv = output_dir / 'summary_statistics_table.csv'
    summary_df.to_csv(summary_csv, index=False)
    print(f"✅ Saved summary table: {summary_csv.name}")
    
    # Create formatted table visualization
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = summary_df[['Task', 'Best_Fold', 'Best_R²', 'Best_r', 'Best_MAE', 'Best_RMSE']].copy()
    table_data.columns = ['Task', 'Best Fold', 'R²', 'r', 'MAE', 'RMSE']
    table_data = table_data.round(3)
    
    table = ax.table(cellText=table_data.values, colLabels=table_data.columns,
                    cellLoc='center', loc='center',
                    bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(table_data.columns)):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight best row (Total_Facial_scale)
    for i, task in enumerate(table_data['Task']):
        if 'Total Facial Scale' in task:
            for j in range(len(table_data.columns)):
                table[(i+1, j)].set_facecolor('#f39c12')
                table[(i+1, j)].set_text_props(weight='bold')
    
    plt.title('Summary Statistics: Best Model Performance per Task', 
             fontsize=16, fontweight='bold', pad=20)
    
    output_path = output_dir / 'table_1_summary_statistics.png'
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_path.name}")
    plt.close()
    
    return summary_df

def main():
    """Main function to generate all visualizations"""
    print("="*80)
    print("GENERATING PUBLICATION-QUALITY VISUALIZATIONS")
    print("="*80)
    
    # Setup paths
    try:
        results_dir = setup_paths()
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease run evaluate_model_v2.py first to generate evaluation results.")
        return
    
    # Create output directory
    output_dir = results_dir / 'visualizations'
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"\n📁 Output directory: {output_dir}")
    
    # Load data
    try:
        results_df, moment_df = load_evaluation_data(results_dir)
        print(f"\n✅ Loaded {len(results_df)} overall results")
        if moment_df is not None:
            print(f"✅ Loaded {len(moment_df)} moment-wise results")
    except Exception as e:
        print(f"\n❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Generate visualizations
    print("\n" + "="*80)
    print("GENERATING FIGURES...")
    print("="*80)
    
    try:
        plot_mae_per_fold(results_df, output_dir)
        plot_r2_per_fold(results_df, output_dir)
        plot_correlation_per_fold(results_df, output_dir)
        plot_moment_wise_mae(moment_df, output_dir)
        plot_moment_wise_comparison(moment_df, output_dir)
        plot_fold_comparison_summary(results_df, output_dir)
        plot_task_performance_comparison(results_df, output_dir)
        plot_predictions_vs_ground_truth(results_df, output_dir)
        summary_df = create_summary_table(results_df, moment_df, output_dir)
        
        print("\n" + "="*80)
        print("✅ ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
        print("="*80)
        print(f"\n📁 All figures saved to: {output_dir}")
        print("\nGenerated figures:")
        print("  - figure_1_mae_per_fold.png")
        print("  - figure_2_r2_per_fold.png")
        print("  - figure_3_correlation_per_fold.png")
        print("  - figure_4_moment_wise_mae_heatmap.png")
        print("  - figure_5_moment_wise_comparison.png")
        print("  - figure_6_fold_comparison_summary.png")
        print("  - figure_7_task_performance_comparison.png")
        print("  - figure_8_predictions_vs_ground_truth.png (if predictions available)")
        print("  - table_1_summary_statistics.png")
        print("  - summary_statistics_table.csv")
        
    except Exception as e:
        print(f"\n❌ Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

