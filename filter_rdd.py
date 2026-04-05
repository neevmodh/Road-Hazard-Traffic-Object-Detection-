import os
import shutil
from collections import defaultdict

# ---------------- PATHS ----------------
base_path = "/Users/lalu/Downloads/datasets"
output_dir = "/Users/lalu/road_project/data/final/images"

os.makedirs(output_dir, exist_ok=True)

# ---------------- SETTINGS ----------------
valid_classes = {0, 1, 2, 3}   # all 4 classes
selected = 0
MAX_IMAGES = 6000

class_count = defaultdict(int)

# ---------------- PROCESS ----------------
for split in ["train", "valid"]:

    label_dir = os.path.join(base_path, split, "labels")
    image_dir = os.path.join(base_path, split, "images")

    for file in os.listdir(label_dir):

        if not file.endswith(".txt"):
            continue

        label_path = os.path.join(label_dir, file)

        keep = False
        present_classes = set()

        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            cls = int(line.split()[0])

            if cls in valid_classes:
                keep = True
                present_classes.add(cls)

        if not keep:
            continue

        img_name = file.replace(".txt", ".jpg")
        img_path = os.path.join(image_dir, img_name)

        if os.path.exists(img_path):
            shutil.copy(img_path, os.path.join(output_dir, img_name))
            selected += 1

            for cls in present_classes:
                class_count[cls] += 1

        if selected >= MAX_IMAGES:
            break

    if selected >= MAX_IMAGES:
        break

# ---------------- RESULT ----------------
print("\nFINAL RESULT (RDD)")
print("Total images:", selected)

print("\nClass distribution:")
for k, v in class_count.items():
    print(f"class {k}:", v)