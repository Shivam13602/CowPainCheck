# GitHub Upload Instructions

## 📤 How to Upload `Ucaps_raw_videos` Folder to GitHub

### **Option 1: Using GitHub Web Interface (Easiest)**

1. **Go to your repository:**
   - Visit: https://github.com/Shivam13602/CowPainCheck

2. **Create the folder:**
   - Click "Add file" → "Create new file"
   - Type: `Ucaps_raw_videos/README.md` (this creates the folder)
   - Copy-paste the README.md content
   - Click "Commit new file"

3. **Upload the Python file:**
   - Click "Add file" → "Upload files"
   - Drag and drop `train_temporal_pain_model_v2.py`
   - Click "Commit changes"

4. **Upload .gitignore:**
   - Click "Add file" → "Create new file"
   - Type: `Ucaps_raw_videos/.gitignore`
   - Copy-paste the .gitignore content
   - Click "Commit new file"

### **Option 2: Using Git Command Line**

```bash
# Navigate to your repository
cd path/to/CowPainCheck

# Copy the folder
cp -r Ucaps_raw_videos ./

# Add files
git add Ucaps_raw_videos/

# Commit
git commit -m "Add UCAPS raw videos training code (v2.0 improved)"

# Push
git push origin main
```

### **Option 3: Using GitHub Desktop**

1. Open GitHub Desktop
2. Select your `CowPainCheck` repository
3. Click "File" → "Add Local Repository" (if needed)
4. Copy the `Ucaps_raw_videos` folder into your repository directory
5. GitHub Desktop will detect the new files
6. Write commit message: "Add UCAPS raw videos training code (v2.0 improved)"
7. Click "Commit to main"
8. Click "Push origin"

## ✅ Verification

After uploading, verify:
- ✅ `Ucaps_raw_videos/train_temporal_pain_model_v2.py` exists
- ✅ `Ucaps_raw_videos/README.md` exists
- ✅ `Ucaps_raw_videos/.gitignore` exists (optional, but recommended)

## 📝 Notes

- The `.gitignore` file prevents large data files from being uploaded
- Only the code is uploaded (not the actual video frames or data)
- The README explains how to use the code in Google Colab

---

**Ready to upload!** 🚀

