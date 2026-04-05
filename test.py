from ultralytics import YOLO
import cv2
import requests
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import defaultdict

# ---------------- MODEL ----------------
model = YOLO("runs/detect/train2/weights/best.pt")

# ---------------- YOUR URLS ----------------
urls = [
    # 🔥 PASTE YOUR 10 IMAGE URLS HERE
    "https://imgs.search.brave.com/rXvRcMXcL0ZCBTDwaFj_YfZT5jmrz-lgYJrpJgTrSwc/rs:fit:500:0:1:0/g:ce/aHR0cHM6Ly9pLnBp/bmltZy5jb20vb3Jp/Z2luYWxzL2E3LzFi/LzMyL2E3MWIzMmQ5/ZDFmMzI1OTA2MzM2/OGZhYmM2YzM3ZDZj/LmpwZw",
    "https://imgs.search.brave.com/Yr18CEd6x2Nn6etJDky6U0MA52QqfhBqIsbdRYHeLfU/rs:fit:500:0:1:0/g:ce/aHR0cHM6Ly93d3cu/c2h1dHRlcnN0b2Nr/LmNvbS9pbWFnZS1w/aG90by9wb3Rob2xl/LXJvYWQtZGFtYWdl/ZC1ydXJhbC1zdXJm/YWNlLTI2MG53LTE5/MjcwODIzNTQuanBn",
    "https://imgs.search.brave.com/3pGmaRkw7OOKlT0azfgHPTz9WOItlcVbE5Obs4a8-qA/rs:fit:500:0:1:0/g:ce/aHR0cHM6Ly9tZWRp/YS5pc3RvY2twaG90/by5jb20vaWQvNTE4/MDIyMjA2L3Bob3Rv/L2NyYWNrZWQtcm9h/ZC1zdXJmYWNlLWFz/cGhhbHQtaW4tYmFk/LWNvbmRpdGlvbnMu/anBnP3M9NjEyeDYx/MiZ3PTAmaz0yMCZj/PWp4d2pHWDUyZmFp/WHdQYTlaeHNoUWJo/TWJUVU4zX1dVaEdJ/aGhrWDhZeTA9",
    "https://imgs.search.brave.com/BPtR-AVtKYbczKZxTumwkqSWiLgs5P2BNlnBmgW-IlQ/rs:fit:500:0:1:0/g:ce/aHR0cHM6Ly9jZG4u/ZGF0YXNldG5pbmph/LmNvbS9wcmV2aWV3/cy9xL2V4dDpqcGVn/L3Jlc2l6ZTpmaWxs/OjQwMDowOjAvcTo3/MC9wbGFpbi9zdXBl/cnZpc2VseS1hc3Nl/dHMtcHVibGljL2lt/YWdlcy9vcmlnaW5h/bC80L1Mvb0QvaVN5/MzFSWmpadGRuOFA3/MXQ4MjlkZVkxdXZt/ZDBCSkE4RGNmUVdy/YlpHS2lINGpiSjhW/RW9tcjlRTFZlVkRa/ZEdZMFY4eVcxVHFX/MGlxbkpPU1RrSEpV/Rm9LN2ZUN2pOV01r/YlJwaEtwbmFqSFpj/b3NFMG81d2dVQ3FB/di5qcGc",
    "https://imgs.search.brave.com/jfGwPfY-Et11w4tVdAHj0yWqLrqgL-LB0LzACY1onnI/rs:fit:500:0:1:0/g:ce/aHR0cHM6Ly90aHVt/YnMuZHJlYW1zdGlt/ZS5jb20vYi9pbmRp/YW4tcm9hZC1oaWdo/d2F5cy1iZWF1dGlm/dWwtbGFuZHNjYXBl/LWluZGlhbi1yb2Fk/cy1iYXJlaWxseS11/dHRhci1wcmFkZXNo/LWluZGlhLWRlY2Vt/YmVyLWluZGlhbi1y/b2FkLWhpZ2h3YXlz/LWJlYXV0aWZ1bC0y/MTY5MDcwNTMuanBn",
    "https://imgs.search.brave.com/cBgLrlP7i2BbB83a5hOTd7PK0HBTlaJyoYLt_V55N0U/rs:fit:500:0:1:0/g:ce/aHR0cHM6Ly93d3cu/Zmh3YS5kb3QuZ292/L3B1YmxpY2F0aW9u/cy9yZXNlYXJjaC9p/bmZyYXN0cnVjdHVy/ZS9wYXZlbWVudHMv/bHRwcC8xMzA5Mi9p/bWFnZXMvaW1hZ2Ux/MDIuanBn",
    "https://imgs.search.brave.com/swYOYwX_s1isaQjzfC1zF845w_odMzavitNyLga9r40/rs:fit:500:0:1:0/g:ce/aHR0cHM6Ly9yZW5v/bGl0aC5jb20uYXUv/d3AtY29udGVudC91/cGxvYWRzLzIwMjQv/MDIvUmVmbGVjdGlv/bi1jcmFja2luZy53/ZWJw",
    "https://imgs.search.brave.com/-BuaKbBbs22Jw0vYboZIShh9Ye-9ErZMhMVx4M7Q9cU/rs:fit:500:0:1:0/g:ce/aHR0cHM6Ly93d3cu/c2lkZW5vdGUubmV3/cy9jb250ZW50L2lt/YWdlcy8yMDIxLzA4/LzgzMzQtVHJhbnN2/ZXJzZS1jb3B5LXNt/LmpwZw",
    "https://imgs.search.brave.com/hmf7ujysQkfxiK4xegkcqAveOD73YCLcTVv2f2fyke4/rs:fit:500:0:1:0/g:ce/aHR0cHM6Ly93d3cu/cGF2ZW1lbnRpbnRl/cmFjdGl2ZS5vcmcv/d3AtY29udGVudC91/cGxvYWRzLzIwMDkv/MDQvQWxsaWdhdG9y/Mi5qcGc",
    "https://imgs.search.brave.com/H7pJXSFQENocJKutoAyqo26HTaS4LllfKojMyZB6S_I/rs:fit:500:0:1:0/g:ce/aHR0cHM6Ly9zdGF0/aWMudmVjdGVlenku/Y29tL3N5c3RlbS9y/ZXNvdXJjZXMvdGh1/bWJuYWlscy8wNDMv/MzA1Lzg1Ni9zbWFs/bC9jYXJzLWRyaXZl/LW9uLWEtaGlnaHdh/eS1waG90by5KUEc"
]

# ---------------- OUTPUT ----------------
os.makedirs("outputs", exist_ok=True)

# ---------------- CLASS NAMES ----------------
names = [
    "pedestrian", "car", "truck", "bus", "motor",
    "longitudinal_crack", "transverse_crack",
    "alligator_crack", "pothole"
]

# ---------------- STATS ----------------
class_counts = defaultdict(int)
risk_counts = defaultdict(int)
total_detections = 0

# ---------------- LOAD IMAGE ----------------
def load_image_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    except:
        return None

# ---------------- PROCESS ----------------
for idx, url in enumerate(urls):

    print("\n" + "="*50)
    print(f"📸 IMAGE {idx+1}")
    print("="*50)

    img = load_image_from_url(url)

    if img is None:
        print("❌ Failed to load image")
        continue

    h, w, _ = img.shape

    def is_on_road(box):
        return ((box[1] + box[3]) / 2) > h * 0.6

    def get_risk(cls, on_road):
        if cls in [5,6,7,8]:
            return "HIGH RISK"
        if on_road:
            if cls == 0:
                return "HIGH RISK"
            if cls in [1,2,3,4]:
                return "MEDIUM RISK"
        return "LOW RISK"

    results = model(img)
    detected = False

    print(f"{'Class':<25} {'Risk':<15}")
    print("-"*40)

    for r in results:
        if r.boxes is None:
            continue

        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()

        if len(boxes) == 0:
            print("⚠️ No detections")

        for box, cls in zip(boxes, classes):
            detected = True
            cls = int(cls)

            risk = get_risk(cls, is_on_road(box))

            print(f"{names[cls]:<25} {risk:<15}")

            # stats
            class_counts[names[cls]] += 1
            risk_counts[risk] += 1
            total_detections += 1

            x1, y1, x2, y2 = map(int, box)

            # color
            color = (0,255,0)
            if risk == "HIGH RISK":
                color = (0,0,255)
            elif risk == "MEDIUM RISK":
                color = (0,165,255)

            label = f"{names[cls]} | {risk}"

            cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
            cv2.putText(img, label, (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # save image
    out_path = f"outputs/output_{idx+1}.jpg"
    cv2.imwrite(out_path, img)

    if not detected:
        print("⚠️ No objects detected")

# ---------------- FINAL SUMMARY ----------------
print("\n" + "="*60)
print("📊 FINAL SUMMARY")
print("="*60)

print(f"Total Images: {len(urls)}")
print(f"Total Detections: {total_detections}\n")

print("🔹 Class Distribution:")
for cls, count in class_counts.items():
    print(f"{cls:<25}: {count}")

print("\n🔹 Risk Distribution:")
for risk, count in risk_counts.items():
    print(f"{risk:<15}: {count}")

# ---------------- GRAPH ----------------
plt.figure()
plt.bar(class_counts.keys(), class_counts.values())
plt.xticks(rotation=45)
plt.title("Class Distribution")
plt.xlabel("Classes")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("class_distribution.png")

plt.figure()
plt.bar(risk_counts.keys(), risk_counts.values())
plt.title("Risk Distribution")
plt.xlabel("Risk Level")
plt.ylabel("Count")
plt.savefig("risk_distribution.png")

print("\n📈 Graphs saved:")
print("✔ class_distribution.png")
print("✔ risk_distribution.png")

print("\n✅ DONE — Outputs + Graphs Ready")