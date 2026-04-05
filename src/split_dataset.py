import json
import random
import os
from collections import defaultdict

# ---------------- PATHS ----------------
input_json = "/Users/lalu/road_project/data/final/annotations.json"
output_dir = "/Users/lalu/road_project/data/final"

with open(input_json) as f:
    coco = json.load(f)

images = coco["images"]
annotations = coco["annotations"]
categories = coco["categories"]

# ---------------- GROUP ANNOTATIONS ----------------
img_to_anns = defaultdict(list)
for ann in annotations:
    img_to_anns[ann["image_id"]].append(ann)

# ---------------- CLASS PRESENCE PER IMAGE ----------------
img_classes = {}

for img in images:
    img_id = img["id"]
    anns = img_to_anns.get(img_id, [])
    classes = set([ann["category_id"] for ann in anns])
    img_classes[img_id] = classes

# ---------------- GROUP BY CLASS ----------------
class_to_images = defaultdict(list)

for img in images:
    img_id = img["id"]
    for cls in img_classes[img_id]:
        class_to_images[cls].append(img)

# ---------------- SPLIT STORAGE ----------------
train_ids = set()
val_ids = set()
test_ids = set()

# ---------------- STRATIFIED SPLIT ----------------
for cls, imgs in class_to_images.items():

    unique_imgs = list({img["id"]: img for img in imgs}.values())
    random.shuffle(unique_imgs)

    n = len(unique_imgs)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    for img in unique_imgs[:n_train]:
        train_ids.add(img["id"])

    for img in unique_imgs[n_train:n_train+n_val]:
        val_ids.add(img["id"])

    for img in unique_imgs[n_train+n_val:]:
        test_ids.add(img["id"])

# ---------------- REMOVE OVERLAP ----------------
val_ids -= train_ids
test_ids -= train_ids
test_ids -= val_ids

# ---------------- CREATE SPLITS ----------------
def create_split(img_ids):
    split_imgs = [img for img in images if img["id"] in img_ids]
    split_anns = [ann for ann in annotations if ann["image_id"] in img_ids]

    return {
        "images": split_imgs,
        "annotations": split_anns,
        "categories": categories
    }

train_data = create_split(train_ids)
val_data = create_split(val_ids)
test_data = create_split(test_ids)

# ---------------- SAVE ----------------
with open(os.path.join(output_dir, "train.json"), "w") as f:
    json.dump(train_data, f)

with open(os.path.join(output_dir, "val.json"), "w") as f:
    json.dump(val_data, f)

with open(os.path.join(output_dir, "test.json"), "w") as f:
    json.dump(test_data, f)

# ---------------- PRINT STATS ----------------
print("\n✅ Dataset split completed!")

print(f"Train images: {len(train_data['images'])}")
print(f"Val images: {len(val_data['images'])}")
print(f"Test images: {len(test_data['images'])}")