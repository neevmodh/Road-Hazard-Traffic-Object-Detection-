import json
import os
import shutil
from collections import defaultdict

# ---------------- PATHS ----------------
label_files = [
    "/Users/lalu/Downloads/archive_g/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json",
    "/Users/lalu/Downloads/archive_g/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json"
]

image_root = "/Users/lalu/Downloads/archive_g/bdd100k/bdd100k/images/100k"
output_dir = "/Users/lalu/road_project/data/final/images"

os.makedirs(output_dir, exist_ok=True)

# ---------------- CLASS FIX ----------------
# Merge bike + motor → motor
def normalize_class(cls):
    if cls in ["motor", "bike"]:
        return "motor"
    return cls

valid_classes = ["person", "car", "truck", "bus", "motor"]

# ---------------- LIMIT ----------------
MAX_IMAGES = 6000

class_count = defaultdict(int)
selected_images = set()
selected = 0

# ---------------- PROCESS ----------------
for label_path in label_files:

    with open(label_path) as f:
        data = json.load(f)

    split = "train" if "train" in label_path else "val"

    for item in data:

        if "labels" not in item:
            continue

        present_classes = set()

        for obj in item["labels"]:
            cls = normalize_class(obj["category"])
            if cls in valid_classes:
                present_classes.add(cls)

        if not present_classes:
            continue

        img_path = os.path.join(image_root, split, item["name"])

        if os.path.exists(img_path) and item["name"] not in selected_images:
            shutil.copy(img_path, os.path.join(output_dir, item["name"]))
            selected_images.add(item["name"])
            selected += 1

            # count classes
            for cls in present_classes:
                class_count[cls] += 1

        if selected >= MAX_IMAGES:
            break

    if selected >= MAX_IMAGES:
        break

# ---------------- RESULT ----------------
print("\nFINAL RESULT")
print("Total images:", selected)

print("\nClass distribution:")
for k, v in class_count.items():
    print(k, ":", v)