import cv2
import numpy as np
import os
import csv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# ---------- Heuristics (unchanged) ----------
def edge_orientation_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    sobelx = cv2.Sobel(edges, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(edges, cv2.CV_64F, 0, 1, ksize=5)
    horizontal_strength = np.sum(np.abs(sobelx))
    vertical_strength = np.sum(np.abs(sobely))
    return (vertical_strength - horizontal_strength) / (vertical_strength + horizontal_strength + 1e-5)

def sky_score(img):
    h, w, _ = img.shape
    region = h // 5
    corners = [
        img[0:region, 0:region],
        img[0:region, -region:],
        img[-region:, 0:region],
        img[-region:, -region:]
    ]
    avg_colors = [np.mean(corner, axis=(0,1)) for corner in corners]
    blue_strength = [c[0] - (c[1]+c[2])/2 for c in avg_colors]
    top_blue = blue_strength[0] + blue_strength[1]
    bottom_blue = blue_strength[2] + blue_strength[3]
    return (top_blue - bottom_blue) / 255.0

def brightness_gradient(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h = gray.shape[0]
    top = np.mean(gray[:h//2])
    bottom = np.mean(gray[h//2:])
    return (top - bottom) / 255.0

def extract_features(img):
    """Extract all three features from an image"""
    e = edge_orientation_score(img)
    s = sky_score(img)
    b = brightness_gradient(img)
    return [e, s, b]

# ---------- Training Module ----------
def train_model_from_csv(csv_path, image_folder):
    """
    Train logistic regression model using labeled data from CSV
    
    Args:
        csv_path: Path to CSV with columns [Filename, ..., PredictedOrientation, ActualOrientation]
        image_folder: Folder containing the images
    
    Returns:
        Trained LogisticRegression model
    """
    print("📊 Loading training data from CSV...")
    
    # Read CSV (last column should be actual label)
    features = []
    labels = []
    filenames = []
    
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        
        for row in reader:
            if not row:
                continue
            filename = row[0]
            actual_orientation = row[-1]  # Last column is ground truth
            
            # Load image
            img_path = os.path.join(image_folder, filename)
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"⚠️  Skipping {filename} (failed to load)")
                continue
            
            # Extract features
            feat = extract_features(img)
            features.append(feat)
            labels.append(1 if actual_orientation == 'Vertical' else 0)
            filenames.append(filename)
    
    features = np.array(features)
    labels = np.array(labels)
    
    print(f"✅ Loaded {len(features)} images")
    print(f"   Vertical: {sum(labels)}, Horizontal: {len(labels) - sum(labels)}")
    
    # Split into train/test
    X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(
        features, labels, filenames, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"\n🔧 Training logistic regression...")
    print(f"   Train: {len(X_train)} images")
    print(f"   Test:  {len(X_test)} images")
    
    # Train model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    # Evaluate
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    print(f"\n📈 Results:")
    print(f"   Training Accuracy:   {train_acc:.2%}")
    print(f"   Test Accuracy:       {test_acc:.2%}")
    
    print(f"\n⚙️  Learned Weights:")
    print(f"   Edge Score:       {model.coef_[0][0]:+.4f}")
    print(f"   Sky Score:        {model.coef_[0][1]:+.4f}")
    print(f"   Brightness Score: {model.coef_[0][2]:+.4f}")
    print(f"   Intercept:        {model.intercept_[0]:+.4f}")
    
    # Detailed test set report
    print(f"\n📋 Test Set Classification Report:")
    print(classification_report(y_test, test_pred, 
                                target_names=['Horizontal', 'Vertical']))
    
    print(f"\n🔢 Confusion Matrix (Test Set):")
    cm = confusion_matrix(y_test, test_pred)
    print(f"                Predicted")
    print(f"              H      V")
    print(f"Actual  H   {cm[0][0]:3d}   {cm[0][1]:3d}")
    print(f"        V   {cm[1][0]:3d}   {cm[1][1]:3d}")
    
    # Show some misclassified examples
    misclassified = [names_test[i] for i in range(len(y_test)) 
                     if y_test[i] != test_pred[i]]
    if misclassified:
        print(f"\n⚠️  Misclassified images ({len(misclassified)}):")
        for name in misclassified[:10]:  # Show first 10
            print(f"   - {name}")
    
    return model

# ---------- Prediction with Trained Model ----------
def predict_with_model(img, model):
    """Use trained model to predict orientation"""
    features = extract_features(img)
    prob = model.predict_proba([features])[0]
    prediction = model.predict([features])[0]
    
    orientation = "Vertical" if prediction == 1 else "Horizontal"
    confidence = prob[prediction]
    
    return orientation, confidence, features

def process_folder_with_model(folder_path, model, output_csv="trained_predictions.csv", rotate=False, confidence_threshold=0.6):
    """Process images using trained model"""
    results = []
    rotated_count = 0
    skipped_low_confidence = 0
    
    # Create rotated folder if needed
    if rotate:
        rotated_folder = os.path.join(folder_path, "rotated")
        os.makedirs(rotated_folder, exist_ok=True)
        print(f"📁 Rotated images will be saved to: {rotated_folder}")
        print(f"🎯 Only rotating images with ≥{confidence_threshold*100:.0f}% confidence\n")
    
    for file in os.listdir(folder_path):
        if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        path = os.path.join(folder_path, file)
        img = cv2.imread(path)
        
        if img is None:
            print(f"❌ Failed to load {file}")
            continue
        
        orientation, confidence, features = predict_with_model(img, model)
        e, s, b = features
        
        # Rotate if image is VERTICAL and confidence is high enough
        if rotate and orientation == "Vertical" and confidence >= confidence_threshold:
            # Rotate 90 degrees clockwise to make it horizontal
            rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            output_path = os.path.join(rotated_folder, file)
            cv2.imwrite(output_path, rotated_img)
            rotated_count += 1
            print(f"{file}: {orientation} → ROTATED ({confidence:.1%} confident) | E={e:.3f}, S={s:.3f}, B={b:.3f}")
        elif rotate and orientation == "Vertical" and confidence < confidence_threshold:
            skipped_low_confidence += 1
            print(f"{file}: {orientation} - SKIPPED (only {confidence:.1%} confident) | E={e:.3f}, S={s:.3f}, B={b:.3f}")
            results.append([file, orientation, confidence, e, s, b, f"Skipped (low confidence)"])
            continue
        else:
            status = "kept as-is" if not rotate else "already horizontal"
            print(f"{file}: {orientation} - {status} ({confidence:.1%} confident) | E={e:.3f}, S={s:.3f}, B={b:.3f}")
        
        results.append([file, orientation, confidence, e, s, b, "Rotated" if (rotate and orientation == "Vertical" and confidence >= confidence_threshold) else "No change"])
    
    # Save results
    csv_path = os.path.join(folder_path, output_csv)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "Orientation", "Confidence", "EdgeScore", "SkyScore", "BrightnessScore", "Action"])
        writer.writerows(results)
    
    print(f"\n✅ Results saved to {csv_path}")
    if rotate:
        print(f"🔄 Rotated {rotated_count} vertical images to horizontal orientation")
        if skipped_low_confidence > 0:
            print(f"⚠️  Skipped {skipped_low_confidence} vertical images due to low confidence (<{confidence_threshold*100:.0f}%)")
        
        # Ask if user wants to overwrite originals
        if rotated_count > 0:
            print(f"\n📁 Rotated images are currently in: {rotated_folder}")
            print("💡 TIP: Review the rotated images first! Delete any incorrectly rotated ones from the folder.")
            print("⚠️  WARNING: The next action will PERMANENTLY replace original files!")
            overwrite = input(f"\nOverwrite originals with rotated versions from folder? (yes/no): ").strip().lower()
            
            if overwrite == "yes":
                import shutil
                
                # Check what's ACTUALLY in the rotated folder now
                rotated_files = [f for f in os.listdir(rotated_folder) 
                                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                if not rotated_files:
                    print("❌ No images found in rotated folder. Nothing to overwrite.")
                    return csv_path
                
                print(f"\n📊 Found {len(rotated_files)} images in rotated folder")
                print(f"   (Originally rotated {rotated_count}, so {rotated_count - len(rotated_files)} were removed)")
                
                confirm = input(f"\nProceed to overwrite {len(rotated_files)} files? (yes/no): ").strip().lower()
                if confirm != "yes":
                    print("✅ Cancelled. Original files kept as-is.")
                    return csv_path
                
                overwritten = 0
                for file in rotated_files:
                    rotated_path = os.path.join(rotated_folder, file)
                    original_path = os.path.join(folder_path, file)
                    try:
                        shutil.copy2(rotated_path, original_path)
                        overwritten += 1
                        print(f"  ✓ Overwrote {file}")
                    except Exception as e:
                        print(f"  ❌ Failed to overwrite {file}: {e}")
                
                print(f"\n✅ Overwrote {overwritten} original files with rotated versions")
                
                # Ask if user wants to delete the rotated folder
                delete_folder = input("\nDelete the 'rotated' folder? (y/n): ").strip().lower()
                if delete_folder == 'y':
                    try:
                        shutil.rmtree(rotated_folder)
                        print(f"🗑️  Deleted {rotated_folder}")
                    except Exception as e:
                        print(f"❌ Failed to delete folder: {e}")
            else:
                print("✅ Original files kept as-is. Rotated versions remain in 'rotated' folder.")
    
    return csv_path

# ---------- Main ----------
if __name__ == "__main__":
    print("🎯 Image Orientation Classifier with Logistic Regression\n")
    
    mode = input("Mode? (1) Train new model, (2) Use pre-trained: ").strip()
    
    if mode == "1":
        # Training mode
        csv_path = input("Enter FULL PATH to labeled CSV file (e.g., C:\\folder\\data.csv): ").strip()
        image_folder = input("Enter folder with training images: ").strip()
        
        if not os.path.exists(csv_path):
            print(f"❌ CSV file not found: {csv_path}")
            print("   Make sure to include the full path with filename!")
        elif not os.path.isdir(image_folder):
            print(f"❌ Image folder not found: {image_folder}")
        else:
            model = train_model_from_csv(csv_path, image_folder)
            
            # Save model
            import pickle
            model_path = os.path.join(os.path.dirname(csv_path), "orientation_model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"\n💾 Model saved to {model_path}")
            
            # Ask if user wants to test on new images
            test = input("\nTest on new images? (y/n): ").strip().lower()
            if test == 'y':
                test_folder = input("Enter test folder path: ").strip()
                if os.path.isdir(test_folder):
                    rotate = input("Rotate vertical photos to horizontal? (y/n): ").strip().lower() == 'y'
                    process_folder_with_model(test_folder, model, rotate=rotate)
    
    elif mode == "2":
        # Prediction mode with existing model
        model_path = input("Enter path to saved model (.pkl): ").strip()
        test_folder = input("Enter folder to classify: ").strip()
        
        if not os.path.exists(model_path):
            print("❌ Model file not found")
        elif not os.path.isdir(test_folder):
            print("❌ Folder not found")
        else:
            import pickle
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print("✅ Model loaded")
            rotate = input("Rotate vertical photos to horizontal? (y/n): ").strip().lower() == 'y'
            process_folder_with_model(test_folder, model, rotate=rotate)
    
    else:
        print("Invalid mode selected")