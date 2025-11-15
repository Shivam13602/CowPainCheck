# Training Recommendations Based on Evaluation Results

## 📊 Current Status Analysis

### **What's Working:**
✅ **Fold 7 is best model** (R²=0.091, MAE=1.912) - Positive R² indicates model is learning  
✅ **Fixed catastrophic failure** (v1.0 R²=-2.015 → v2.0 R²=0.091)  
✅ **Strong individual features** (Ears_frontal r=0.628, Orbital_tightening r=0.562)  
✅ **Good baseline performance** (M0 MAE=0.784, M4 MAE=0.940)

### **Critical Issues:**
❌ **6 out of 9 folds stopped at epoch 1** - Severely under-trained  
❌ **M2 (peak pain) performance poor** (MAE=3.570, 4.5× worse than M0)  
❌ **Low R² overall** (0.091 = only 9% variance explained)  
❌ **Some features have negative R²** despite positive correlations

## 🎯 Recommended Action Plan

### **PHASE 1: Test Current Best Model (Fold 7) - DO THIS FIRST**

**Why:** Before retraining, verify if current model is usable for your application.

**Actions:**
1. ✅ **Deploy Fold 7** (`best_model_v2_fold_7.pt`) for testing
2. ✅ **Test on real-world data** (if available)
3. ✅ **Evaluate clinical utility** - Is MAE=1.912 acceptable for your use case?
4. ✅ **Check M2 performance** - Can you accept MAE=3.570 for acute pain detection?

**Decision Point:**
- **If acceptable:** Use current model, focus on deployment
- **If not acceptable:** Proceed to Phase 2 (retraining)

---

### **PHASE 2: Retrain with Improved Hyperparameters - RECOMMENDED**

**Why:** Current model is under-trained (6 folds at epoch 1) and R² is low (0.091).

## 🔧 Recommended Training Improvements

### **1. Fix Early Stopping (CRITICAL)**

**Current Issue:** 6/9 folds stopped at epoch 1 - too aggressive

**Recommended Changes:**
```python
config = {
    'patience': 15,           # Increase from 10 to 15
    'min_delta': 0.0001,     # Add minimum improvement threshold (was None)
    'min_epochs': 5,         # NEW: Force minimum 5 epochs before early stopping
}
```

**Rationale:**
- Prevents premature stopping
- Allows model to learn for at least 5 epochs
- More patience gives model time to converge

### **2. Lower Learning Rate (IMPORTANT)**

**Current:** `learning_rate: 0.0001`

**Recommended:**
```python
config = {
    'learning_rate': 0.00005,  # Half the current rate (5e-5)
    # OR even lower:
    # 'learning_rate': 0.00003,  # 3e-5 for more stable training
}
```

**Rationale:**
- Lower LR = more stable training
- Better convergence for complex multi-task learning
- Reduces risk of overshooting optimal weights
- Fold 7 trained 17 epochs - lower LR might help it converge better

### **3. Increase Weight Decay (IMPORTANT)**

**Current:** `weight_decay: 1e-5`

**Recommended:**
```python
config = {
    'weight_decay': 1e-4,  # 10× increase (0.0001)
    # OR moderate:
    # 'weight_decay': 5e-5,  # 5× increase
}
```

**Rationale:**
- Stronger regularization prevents overfitting
- Helps with negative R² issues (model learning noise)
- Better generalization
- Fold 3 overfitted (29 epochs, R²=-0.350) - more weight decay helps

### **4. Enhanced Data Augmentation (RECOMMENDED)**

**Current Augmentation:**
```python
transforms.RandomHorizontalFlip(p=0.5)
transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
transforms.RandomRotation(degrees=10)
```

**Recommended Enhanced Augmentation:**
```python
augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(
        brightness=0.3,      # Increase from 0.2
        contrast=0.3,        # Increase from 0.2
        saturation=0.3,      # Increase from 0.2
        hue=0.1              # NEW: Add hue jitter
    ),
    transforms.RandomRotation(degrees=15),  # Increase from 10
    transforms.RandomAffine(                 # NEW: Add affine transforms
        degrees=0,
        translate=(0.1, 0.1),  # 10% translation
        scale=(0.9, 1.1)        # 10% scaling
    ),
    transforms.GaussianBlur(                # NEW: Add blur (optional)
        kernel_size=3,
        sigma=(0.1, 0.5)
    ),
])
```

**Rationale:**
- More data diversity = better generalization
- Helps with M2 challenge (more varied pain expressions)
- Reduces overfitting risk
- Improves robustness to lighting/angle variations

### **5. Learning Rate Scheduling Improvements**

**Current:** `ReduceLROnPlateau` with `factor=0.5, patience=5`

**Recommended:**
```python
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=7,        # Increase from 5
    min_lr=1e-7,
    verbose=True      # Add verbose for monitoring
)
```

**OR use Cosine Annealing:**
```python
from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler = CosineAnnealingLR(
    optimizer,
    T_max=config['num_epochs'],  # 50 epochs
    eta_min=1e-7
)
```

**Rationale:**
- Cosine annealing provides smoother LR decay
- Better for long training runs
- Helps model converge to better minima

### **6. Gradient Clipping Adjustment**

**Current:** `max_norm=1.0`

**Recommended:**
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # Tighter clipping
```

**Rationale:**
- Tighter clipping = more stable training
- Prevents large gradient updates
- Better for lower learning rates

### **7. Batch Size (Keep Current)**

**Current:** `batch_size=64` (T4 GPU)

**Recommendation:** ✅ **Keep as is** - Already optimized for T4

---

## 📋 Complete Recommended Config

```python
config = {
    # Hardware
    'batch_size': 64,           # Keep (T4 optimized)
    'num_workers': 0,          # Keep (Colab)
    
    # Model Architecture
    'max_frames': 32,          # Keep
    'resolution': (112, 112),  # Keep
    'lstm_hidden_size': 256,   # Keep
    'dropout_rate': 0.5,       # Keep
    
    # Training Hyperparameters - IMPROVED
    'num_epochs': 50,           # Keep
    'learning_rate': 0.00005,  # ⬇️ REDUCED (was 0.0001)
    'weight_decay': 1e-4,      # ⬆️ INCREASED (was 1e-5)
    'gradient_clip': 0.5,      # ⬇️ TIGHTER (was 1.0)
    
    # Early Stopping - IMPROVED
    'patience': 15,            # ⬆️ INCREASED (was 10)
    'min_delta': 0.0001,      # ➕ NEW (prevents tiny improvements)
    'min_epochs': 5,          # ➕ NEW (force minimum training)
    
    # Learning Rate Scheduling
    'lr_factor': 0.5,         # Keep
    'lr_patience': 7,         # ⬆️ INCREASED (was 5)
    'min_lr': 1e-7,           # Keep
}
```

---

## 🎯 Training Strategy

### **Option A: Retrain All Folds (Recommended)**

**Steps:**
1. Use improved config above
2. Retrain all 9 folds from scratch
3. Compare results with current Fold 7
4. Select best model

**Time:** ~9-14 hours (T4 GPU)

**Pros:**
- Fresh start with better hyperparameters
- All folds properly trained
- Fair comparison

**Cons:**
- Time-consuming
- May not improve if data is the limiting factor

### **Option B: Continue Training Under-trained Folds**

**Steps:**
1. Load checkpoints for folds that stopped at epoch 1
2. Continue training with improved config
3. Keep Fold 7 as baseline

**Time:** ~2-3 hours (only 6 folds)

**Pros:**
- Faster
- Builds on existing training
- Preserves Fold 7

**Cons:**
- May inherit issues from epoch 1 initialization
- Not a clean comparison

### **Option C: Fine-tune Best Model (Fold 7)**

**Steps:**
1. Load Fold 7 checkpoint
2. Fine-tune with lower LR (1e-5) for 10-20 epochs
3. Focus on M2 performance

**Time:** ~1-2 hours

**Pros:**
- Fastest option
- Builds on best model
- Can target M2 improvement

**Cons:**
- May not address fundamental issues
- Limited improvement potential

---

## 🎯 My Recommendation: **Option A (Retrain All Folds)**

### **Why:**
1. **6/9 folds are severely under-trained** - Need proper training
2. **Low R² (0.091)** - Room for significant improvement
3. **M2 challenge** - Better training might help
4. **Clean comparison** - Fair evaluation of improvements

### **Expected Improvements:**
- ✅ All folds train for proper duration (5+ epochs minimum)
- ✅ Better convergence with lower LR
- ✅ Reduced overfitting with higher weight decay
- ✅ Better generalization with enhanced augmentation
- ✅ Improved M2 performance (hopefully)
- ✅ Higher R² scores (target: 0.15-0.25)

### **Success Criteria:**
- **Minimum:** R² > 0.15 (vs current 0.091)
- **Target:** R² > 0.20
- **M2 MAE:** < 3.0 (vs current 3.570)
- **All folds:** Train for at least 5 epochs

---

## 📝 Implementation Checklist

Before retraining:

- [ ] **Backup current models** (save to separate folder)
- [ ] **Update training script** with improved config
- [ ] **Test config on 1 fold first** (Fold 0) to verify it works
- [ ] **Monitor first epoch** - check loss is reasonable
- [ ] **Verify early stopping** - ensure it doesn't trigger too early
- [ ] **Check GPU memory** - ensure batch_size=64 still works
- [ ] **Run full training** (all 9 folds)
- [ ] **Compare results** with current Fold 7
- [ ] **Select best model** for deployment

---

## 🚨 Important Notes

1. **Test Current Model First:** Before retraining, test Fold 7 to see if it meets your needs
2. **Incremental Changes:** Don't change everything at once - test one improvement at a time
3. **Monitor Training:** Watch for overfitting (validation loss increasing)
4. **M2 Challenge:** May need specialized approach (M2-specific augmentation or loss weighting)
5. **Data Quality:** If R² stays low, consider if data quality/quantity is the issue

---

## 🔬 Alternative: M2 Specialist Model

If M2 performance remains poor after retraining, consider:

1. **M2-specific augmentation** (more aggressive for M2 samples)
2. **M2 specialist model** (separate model trained only on M2 data)
3. **Ensemble approach** (combine general model + M2 specialist)
4. **Higher M2 weight** (increase from 2.5× to 3.5× or 4.0×)

---

## 📊 Expected Timeline

- **Phase 1 (Testing):** 1-2 days
- **Phase 2 (Retraining):** 1-2 days (9-14 hours training + evaluation)
- **Total:** 2-4 days

---

**Bottom Line:** Test current Fold 7 first. If not acceptable, retrain all folds with improved hyperparameters (lower LR, higher weight decay, better augmentation, fixed early stopping).

