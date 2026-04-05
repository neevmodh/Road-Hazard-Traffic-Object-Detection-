import os
import shutil
import random

BASE = "/Users/lalu/road_project/data/yolo_balanced"

IMG_DIR = os.path.join(BASE, "images/train")
LBL_DIR = os.path.join(BASE, "labels/train")

# weak classes
WEAK_CLASSES = [4, 5, 6, 7, 8]   # motor + cracks + pothole

images = os.listdir(IMG_DIR)

boosted = 0

for img in images:
    label_path = os.path.join(LBL_DIR, img.replace(".jpg", ".txt"))
    if not os.path.exists(label_path):
        continue

    with open(label_path) as f:
        classes = [int(line.split()[0]) for line in f.readlines()]

    if any(c in WEAK_CLASSES for c in classes):
        # duplicate image
        new_name = f"boost_{random.randint(0,999999)}_{img}"

        shutil.copy(
            os.path.join(IMG_DIR, img),
            os.path.join(IMG_DIR, new_name)
        )

        shutil.copy(
            label_path,
            os.path.join(LBL_DIR, new_name.replace(".jpg", ".txt"))
        )

        boosted += 1

print("✅ Boosted images:", boosted)