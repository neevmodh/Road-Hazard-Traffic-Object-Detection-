import os
import random
import shutil

# -------- PATHS --------
BASE_PATH = "/Users/lalu/road_project/data/yolo"

IMG_TRAIN = os.path.join(BASE_PATH, "images/train")
IMG_VAL = os.path.join(BASE_PATH, "images/val")

LBL_TRAIN = os.path.join(BASE_PATH, "labels/train")
LBL_VAL = os.path.join(BASE_PATH, "labels/val")

# -------- CREATE VAL FOLDERS --------
os.makedirs(IMG_VAL, exist_ok=True)
os.makedirs(LBL_VAL, exist_ok=True)

# -------- LOAD FILES --------
images = [f for f in os.listdir(IMG_TRAIN) if f.endswith(".jpg")]

# -------- SHUFFLE --------
random.shuffle(images)

# -------- SPLIT (20%) --------
split_size = int(0.2 * len(images))
val_images = images[:split_size]

print(f"Total images: {len(images)}")
print(f"Moving to val: {len(val_images)}")

# -------- MOVE FILES --------
moved = 0

for img in val_images:
    img_src = os.path.join(IMG_TRAIN, img)
    img_dst = os.path.join(IMG_VAL, img)

    label_name = img.replace(".jpg", ".txt")
    lbl_src = os.path.join(LBL_TRAIN, label_name)
    lbl_dst = os.path.join(LBL_VAL, label_name)

    # Move image
    if os.path.exists(img_src):
        shutil.move(img_src, img_dst)

    # Move label (if exists)
    if os.path.exists(lbl_src):
        shutil.move(lbl_src, lbl_dst)

    moved += 1

print(f"✅ Successfully moved {moved} images to validation set")