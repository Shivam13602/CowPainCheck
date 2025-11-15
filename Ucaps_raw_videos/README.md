# UCAPS Raw Videos - Temporal Pain Model Training

This folder contains the training code for the automated cattle pain detection model (v2.0).

## 📁 Files

- `train_temporal_pain_model_v2.py` - Complete training script for Google Colab

## 🎯 Key Features

### **Model Architecture:**
- 3D CNN + Bidirectional LSTM + Attention Mechanism
- 7 individual facial feature outputs
- **Total Facial Scale calculated from 7 features** (UNESP-Botucatu validated method)

### **Loss Function:**
- **Feature-weighted loss**: Orbital/Ears get higher weight (1.37-1.56×) based on correlation with pain
- **Moment-weighted loss**: M2 (acute pain) gets 2.5× weight
- **Consistency loss**: Ensures Total predicted ≈ Total calculated

### **Improvements:**
- ✅ Total Facial Scale = sum(7 features) (peer-reviewed method)
- ✅ Feature weights based on correlation analysis
- ✅ Fixed label mapping (Total.Facial.scale ↔ Total_Facial_scale)
- ✅ Fully resumable (handles Colab disconnects)
- ✅ Optimized for T4 GPU (batch_size=32)

## 📊 Total Facial Scale Mechanism

Based on **UNESP-Botucatu Cattle Pain Scale** (validated in peer-reviewed journals):

**Total Facial Scale = Sum of 7 Individual Features**

- Each feature: 0-2 scale
- Total: 0-14 scale
- This is the **validated calculation method** from veterinary literature

The model:
1. Predicts 7 individual features
2. **Calculates** Total = sum(7 features) ← Primary output (validated)
3. Also predicts Total directly (for consistency loss)
4. Enforces consistency: predicted ≈ calculated

## 🚀 Usage

### **For Google Colab:**

1. Upload this folder to Google Drive
2. Open `train_temporal_pain_model_v2.py` in Colab
3. Make sure you have:
   - `train_val_test_splits_v2.json`
   - `sequence_label_mapping_v2.json`
   - `sequence/` directory with video frames
4. Run the script
5. It will ask which fold to start from (0-8)

### **Requirements:**
- PyTorch 2.0+
- Google Colab Pro (T4 GPU recommended)
- Data files in `/content/drive/MyDrive/facial_pain_project_v2/`

## 📝 Model Outputs

The model outputs:
- 7 individual facial features (0-2 each)
- **Total_Facial_scale** (0-14, calculated from 7 features)
- Attention weights (for interpretability)

## 🔬 Scientific Foundation

- **UNESP-Botucatu Cattle Pain Scale**: Validated veterinary pain assessment tool
- **Feature correlations**: Based on statistical analysis of 300 expert evaluations
- **Total calculation**: Peer-reviewed method from veterinary literature

## 📚 References

- UNESP-Botucatu Cattle Pain Scale validation studies
- Veterinary pain assessment literature
- Deep learning for animal welfare applications

---

**Version:** v2.0 Improved  
**Last Updated:** November 2024  
**Repository:** https://github.com/Shivam13602/CowPainCheck

