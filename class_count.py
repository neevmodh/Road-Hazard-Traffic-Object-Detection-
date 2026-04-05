import json
from collections import defaultdict

label_files = [
    "/Users/lalu/Downloads/archive_g/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json",
    "/Users/lalu/Downloads/archive_g/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json"
]

class_count = defaultdict(int)
image_count = 0

for label_path in label_files:
    with open(label_path) as f:
        data = json.load(f)

    for item in data:
        if "labels" not in item:
            continue

        image_count += 1

        for obj in item["labels"]:
            class_count[obj["category"]] += 1

# print results
print("\nTOTAL IMAGES:", image_count)
print("\nCLASS DISTRIBUTION (OBJECT COUNT):")

for k, v in sorted(class_count.items(), key=lambda x: x[1], reverse=True):
    print(k, ":", v)