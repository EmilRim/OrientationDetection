import os
import csv
from PIL import Image

def main():
    # === Ask for the image folder ===
    image_folder = input("Enter the path to your image folder: ").strip()
    if not os.path.isdir(image_folder):
        print("❌ That’s not a valid folder. Try again.")
        return

    output_csv = os.path.join(image_folder, "labels.csv")

    # === Load existing labels if resuming ===
    labels = {}
    if os.path.exists(output_csv):
        with open(output_csv, newline='', encoding='utf-8') as f:
            for row in csv.reader(f):
                if row:
                    labels[row[0]] = row[1]

    # === Collect images ===
    valid_exts = ('.jpg', '.jpeg', '.png')
    files = [f for f in sorted(os.listdir(image_folder)) if f.lower().endswith(valid_exts)]

    print(f"\nFound {len(files)} images. Already labeled {len(labels)}.\n")

    # === Label loop ===
    for fname in files:
        if fname in labels:
            continue  # skip already labeled

        path = os.path.join(image_folder, fname)
        try:
            img = Image.open(path)
            img.show()
        except Exception as e:
            print(f"Error opening {fname}: {e}")
            continue

        while True:
            label = input(f"{fname} — is it vertical (v) or horizontal (h)? ").strip().lower()
            if label in ("v", "h"):
                labels[fname] = "Vertical" if label == "v" else "Horizontal"
                break
            else:
                print("Type 'v' or 'h', not whatever nonsense that was.")

        # === Save progress immediately ===
        with open(output_csv, "w", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for k, v in labels.items():
                writer.writerow([k, v])

        img.close()

        # Close default image viewer (Windows only)
        os.system("taskkill /im Microsoft.Photos.exe /f >nul 2>&1")
        print(f"✅ Saved: {fname} = {labels[fname]}\n")

    print("\n🎉 All images labeled. Now go train your model like a civilized human.")

if __name__ == "__main__":
    main()
