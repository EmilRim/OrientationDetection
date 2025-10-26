# Orientation Detection 

Train a machine learning model to detect and fix image orientation using logistic regression.

## 🚀 Features

- **Smart Detection**: Uses edge analysis, sky detection, and brightness gradients
- **Trained Model**: Learns optimal weights from your labeled data (85-92% accuracy)
- **Auto-Rotation**: Rotates vertical photos to horizontal (or vice versa)
- **Confidence Threshold**: Only rotates images with ≥60% confidence
- **Safe Workflow**: Review rotated images before overwriting originals

## 📋 Requirements

```bash
pip install opencv-python numpy scikit-learn pillow
```

## 🎯 Quick Start

### 0️⃣ Label Your Training Data (Optional)

If you don't have labeled data yet, use the included labeling tool:

```bash
python label_images.py
# Provide: Folder with images to label
```

**Interactive Workflow:**
1. Script opens each image automatically
2. Type `v` (vertical) or `h` (horizontal)
3. Progress saved after each image
4. Resume anytime if interrupted
5. Outputs `labels.csv` ready for training

### 1️⃣ Train Your Model

```bash
python detect_orientation.py
# Choose: 1 (Train new model)
# Provide: CSV path with labeled data
# Provide: Folder with training images
```

**CSV Format** (created by labeling tool or manually):
```csv
Filename,ActualOrientation
DSCF9425.JPG,Horizontal
DSCF9429.JPG,Vertical
...
```

Or with full feature scores:
```csv
Filename,TotalScore,EdgeScore,SkyScore,BrightnessScore,PredictedOrientation,ActualOrientation
DSCF9425.JPG,-0.136,-0.316,0.526,-0.258,Horizontal,Horizontal
DSCF9429.JPG,0.079,0.190,0.066,-0.039,Vertical,Vertical
...
```

*(Both formats work - the script uses the last column as ground truth)*

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
├── image1.jpg               # Original images
├── image2.jpg
├── labels.csv               # Created by labeling tool
├── rotated/                 # Rotated versions (auto-created)
│   ├── image1.jpg
│   └── image2.jpg
├── orientation_model.pkl    # Saved ML model
└── trained_predictions.csv  # Classification results
```

## 📝 Included Scripts

### `label_images.py`
Interactive labeling tool for creating training data.

**Features:**
- Auto-opens each image in system viewer
- Simple `v`/`h` keyboard input
- Auto-saves progress after each label
- Resume from where you left off
- Auto-closes image viewer (Windows)
- Creates `labels.csv` for training

**Usage:**
```bash
python label_images.py
Enter the path to your image folder: C:\Photos
```

### `detect_orientation.py`
Main classifier - trains models and rotates images.

**Mode 1:** Train from labeled data  
**Mode 2:** Classify with trained model

## 🛡️ Safety Features

- 💾 Preserves image metadata (EXIF, dates)
- 📁 Rotated images saved separately first
- 🔄 Dynamic folder checking before overwrite
- 🗑️ Optional cleanup of `rotated/` folder

## 💡 Tips

- **Label 100+ images** for best training results (labeling tool makes this fast!)
- **Mix of both orientations** - aim for roughly 50/50 split
- **Review low-confidence images** (<65%) manually
- **Keep `rotated/` folder** until you're 100% satisfied
- **Backup originals** before first use!
- **Labeling tool auto-saves** - you can quit and resume anytime

## 🔄 Complete Workflow

1. **Label** → Run `label_images.py` on 100-200 images
2. **Train** → Run `detect_orientation.py` mode 1 with `labels.csv`
3. **Test** → Check accuracy on test set (script shows this)
4. **Deploy** → Run mode 2 on new folders
5. **Review** → Check `rotated/` folder, delete mistakes
6. **Commit** → Overwrite originals (only files in folder are touched)
