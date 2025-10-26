# OrientationDetection

Train a machine learning model to detect and fix image orientation using logistic regression.

## 🚀 Features

- **Smart Detection**: Uses edge analysis, sky detection, and brightness gradients
- **Trained Model**: Learns optimal weights from your labeled data (85-92% accuracy)
- **Auto-Rotation**: Rotates vertical photos to horizontal (or vice versa)
- **Confidence Threshold**: Only rotates images with ≥65% confidence
- **Safe Workflow**: Review rotated images before overwriting originals

## 📋 Requirements

```bash
pip install opencv-python numpy scikit-learn
```

## 🎯 Quick Start

### 1️⃣ Train Your Model

```bash
python detect_orientation.py
# Choose: 1 (Train new model)
# Provide: CSV path with labeled data
# Provide: Folder with training images
```

**CSV Format:**
```csv
Filename,TotalScore,EdgeScore,SkyScore,BrightnessScore,PredictedOrientation,ActualOrientation
DSCF9425.JPG,-0.136,-0.316,0.526,-0.258,Horizontal,Horizontal
DSCF9429.JPG,0.079,0.190,0.066,-0.039,Vertical,Vertical
...
```

The script will:
- Train on 80% of your data
- Test on 20%
- Save model as `orientation_model.pkl`
- Show accuracy and learned weights

### 2️⃣ Classify & Rotate New Images

```bash
python detect_orientation.py
# Choose: 2 (Use pre-trained model)
# Provide: Path to orientation_model.pkl
# Provide: Folder to classify
# Choose: yes (to rotate vertical → horizontal)
```

### 3️⃣ Review & Overwrite

1. **Check the `rotated/` folder** - Review all rotated images
2. **Delete any mistakes** - Remove incorrectly rotated photos
3. **Return to script** - Type `yes` to overwrite originals
4. **Only files in `rotated/` are overwritten** - Deleted ones are skipped!

## 🔧 How It Works

The model analyzes three features:

- **Edge Score**: Vertical vs horizontal edge strength (Sobel filters)
- **Sky Score**: Blue color in top vs bottom corners
- **Brightness Score**: Top vs bottom brightness gradient

Logistic regression learns optimal weights from your labeled examples.

## 📊 Example Output

```
📊 Found 142 images in rotated folder
   (Originally rotated 150, so 8 were removed)

Proceed to overwrite 142 files? (yes/no): yes

✅ Overwrote 142 original files with rotated versions
```

## ⚙️ Configuration

**Confidence Threshold**: Edit line 90 in the script
```python
confidence_threshold=0.65  # 65% minimum (0.0 to 1.0)
```

**Rotation Direction**: Currently rotates vertical → horizontal. To reverse:
- Change line 30: `if orientation == "Vertical"` → `"Horizontal"`

## 📁 File Structure

```
your_folder/
├── image1.jpg           # Original images
├── image2.jpg
├── rotated/             # Rotated versions (created automatically)
│   ├── image1.jpg
│   └── image2.jpg
├── orientation_model.pkl    # Saved model
└── trained_predictions.csv  # Results log
```

## 🛡️ Safety Features

- 💾 Preserves image metadata (EXIF, dates)
- 📁 Rotated images saved separately first
- 🔄 Dynamic folder checking before overwrite
- 🗑️ Optional cleanup of `rotated/` folder

## 💡 Tips

- **Use 100+ labeled images** for best training results
- **Review low-confidence images** (<65%) manually
- **Keep `rotated/` folder** until you're 100% satisfied
- **Backup originals** before first use!
