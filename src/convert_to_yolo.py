import json
import os
from PIL import Image

# -------- PATHS --------
json_file = "/Users/lalu/road_project/data/final/train.json"
images_dir = "/Users/lalu/road_project/data/final/images"
labels_dir = "/Users/lalu/road_project/data/yolo/labels/train"

os.makedirs(labels_dir, exist_ok=True)

# -------- LOAD COCO --------
with open(json_file) as f:
    coco = json.load(f)

images = {img["id"]: img for img in coco["images"]}
annotations = coco["annotations"]

# -------- GROUP ANNOTATIONS --------
img_to_anns = {}
for ann in annotations:
    img_to_anns.setdefault(ann["image_id"], []).append(ann)

# -------- CONVERT --------
for img_id, img in images.items():
    file_name = img["file_name"]
    width = img["width"]
    height = img["height"]

    label_path = os.path.join(labels_dir, file_name.replace(".jpg", ".txt"))

    with open(label_path, "w") as f:
        for ann in img_to_anns.get(img_id, []):
            x, y, w, h = ann["bbox"]

            # YOLO format (normalized)
            x_center = (x + w / 2) / width
            y_center = (y + h / 2) / height
            w /= width
            h /= height

            cls = ann["category_id"]

            f.write(f"{cls} {x_center} {y_center} {w} {h}\n")

print("✅ COCO → YOLO conversion done")