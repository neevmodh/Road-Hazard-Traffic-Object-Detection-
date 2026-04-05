import os
import random
import shutil
from collections import defaultdict

# -------- PATHS --------
BASE = "/Users/lalu/road_project/data/yolo"

IMG_SRC = os.path.join(BASE, "images/train")
LBL_SRC = os.path.join(BASE, "labels/train")

IMG_DST = os.path.join(BASE, "images_balanced/train")
LBL_DST = os.path.join(BASE, "labels_balanced/train")

os.makedirs(IMG_DST, exist_ok=True)
os.makedirs(LBL_DST, exist_ok=True)

# -------- SETTINGS --------
TARGET_TOTAL = 4000
TARGET_PER_CLASS = 500   # 🔥 balance control

# -------- LOAD FILES --------
images = [f for f in os.listdir(IMG_SRC) if f.endswith(".jpg")]
random.shuffle(images)

# -------- CLASS COUNTS --------
class_count = defaultdict(int)
selected = set()

# -------- STEP 1: STRONG BALANCE --------
for img in images:
    if len(selected) >= TARGET_TOTAL:
        break

    label_path = os.path.join(LBL_SRC, img.replace(".jpg", ".txt"))
    if not os.path.exists(label_path):
        continue

    with open(label_path) as f:
        classes = [int(line.split()[0]) for line in f.readlines()]

    # pick image if it helps any class
    pick = False
    for c in classes:
        if class_count[c] < TARGET_PER_CLASS:
            pick = True
            break

    if pick:
        selected.add(img)
        for c in classes:
            class_count[c] += 1

# -------- STEP 2: FILL REMAINING --------
for img in images:
    if len(selected) >= TARGET_TOTAL:
        break
    if img not in selected:
        selected.add(img)

# -------- COPY FILES --------
for img in selected:
    shutil.copy(
        os.path.join(IMG_SRC, img),
        os.path.join(IMG_DST, img)
    )

    label_src = os.path.join(LBL_SRC, img.replace(".jpg", ".txt"))
    label_dst = os.path.join(LBL_DST, img.replace(".jpg", ".txt"))

    if os.path.exists(label_src):
        shutil.copy(label_src, label_dst)

# -------- PRINT RESULT --------
print("\n✅ DONE")
print("Total images:", len(selected))
print("Class distribution:")
for k in sorted(class_count.keys()):
    print(f"class {k}: {class_count[k]}")