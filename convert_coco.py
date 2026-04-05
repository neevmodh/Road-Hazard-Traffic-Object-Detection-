import os
import json
from tqdm import tqdm

# ---------------- PATHS ----------------
final_img_dir = "/Users/lalu/road_project/data/final/images"
output_json = "/Users/lalu/road_project/data/final/annotations.json"

bdd_labels = [
    "/Users/lalu/Downloads/archive_g/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json",
    "/Users/lalu/Downloads/archive_g/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json"
]

rdd_base = "/Users/lalu/Downloads/datasets"

# ---------------- CLASS MAP ----------------
bdd_map = {
    "person": 0,
    "car": 1,
    "truck": 2,
    "bus": 3,
    "motor": 4,
    "bike": 4
}

rdd_map = {
    0: 5,  # D00
    1: 6,  # D10
    2: 7,  # D20
    3: 8   # D40
}

# ---------------- INIT COCO ----------------
coco = {
    "images": [],
    "annotations": [],
    "categories": [
        {"id": 0, "name": "pedestrian"},
        {"id": 1, "name": "car"},
        {"id": 2, "name": "truck"},
        {"id": 3, "name": "bus"},
        {"id": 4, "name": "motor"},
        {"id": 5, "name": "longitudinal_crack"},
        {"id": 6, "name": "transverse_crack"},
        {"id": 7, "name": "alligator_crack"},
        {"id": 8, "name": "pothole"}
    ]
}

img_id = 0
ann_id = 0

# ---------------- BDD → COCO ----------------
for label_file in bdd_labels:
    with open(label_file) as f:
        data = json.load(f)

    for item in tqdm(data, desc="BDD Processing"):

        img_name = item["name"]
        img_path = os.path.join(final_img_dir, img_name)

        if not os.path.exists(img_path):
            continue

        coco["images"].append({
            "id": img_id,
            "file_name": img_name,
            "width": 1280,
            "height": 720
        })

        if "labels" in item:
            for obj in item["labels"]:
                cat = obj["category"]

                if cat not in bdd_map:
                    continue

                box = obj["box2d"]
                x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]

                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": bdd_map[cat],
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": (x2 - x1) * (y2 - y1),
                    "iscrowd": 0
                })

                ann_id += 1

        img_id += 1

# ---------------- RDD → COCO ----------------
for split in ["train", "valid"]:
    label_dir = os.path.join(rdd_base, split, "labels")

    for file in tqdm(os.listdir(label_dir), desc=f"RDD {split}"):

        if not file.endswith(".txt"):
            continue

        img_name = file.replace(".txt", ".jpg")
        img_path = os.path.join(final_img_dir, img_name)

        if not os.path.exists(img_path):
            continue

        coco["images"].append({
            "id": img_id,
            "file_name": img_name,
            "width": 600,
            "height": 600
        })

        with open(os.path.join(label_dir, file)) as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            cls = int(parts[0])

            if cls not in rdd_map:
                continue

            x, y, w, h = map(float, parts[1:])

            # convert YOLO → COCO
            x1 = (x - w/2) * 600
            y1 = (y - h/2) * 600
            bw = w * 600
            bh = h * 600

            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": rdd_map[cls],
                "bbox": [x1, y1, bw, bh],
                "area": bw * bh,
                "iscrowd": 0
            })

            ann_id += 1

        img_id += 1

# ---------------- SAVE ----------------
with open(output_json, "w") as f:
    json.dump(coco, f)

print("\n✅ COCO Conversion Done!")
print("Total images:", len(coco["images"]))
print("Total annotations:", len(coco["annotations"]))