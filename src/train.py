from ultralytics import YOLO

# ✅ Best model for your system
model = YOLO("yolov8s.pt")

model.train(
    data="/Users/lalu/road_project/data/yolo.yaml",

    # ---------------- TRAINING ----------------
    epochs=50,
    imgsz=416,
    batch=8,
    device="cpu",
    workers=4,          # 🔥 TRY this (may fallback to 0)

    # ---------------- OPTIMIZATION ----------------
    optimizer="SGD",
    lr0=0.001,
    momentum=0.937,
    weight_decay=0.0005,

    # ---------------- AUGMENTATION ----------------
    augment=True,
    mosaic=0.3,
    mixup=0.0,
    copy_paste=0.0,

    # ---------------- TRAINING STRATEGY ----------------
    cos_lr=True,
    patience=3,

    # ---------------- PERFORMANCE ----------------
    cache="disk",
    amp=False,

    # ---------------- SAVE ----------------
    project="runs",
    name="final_model_workers4",
    exist_ok=True,
    save=True,

    # ---------------- VALIDATION ----------------
    val=True,
    plots=True,
    verbose=True
)

print("✅ FINAL TRAINING COMPLETE (workers=4)")