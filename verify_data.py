import os
from PIL import Image

bad = total = 0
for cls in os.listdir("data/raw"):
    d = os.path.join("data/raw", cls)
    if not os.path.isdir(d): continue
    for f in os.listdir(d):
        if not f.lower().endswith(('.jpg','.jpeg','.png')): continue
        total += 1
        try:
            with Image.open(os.path.join(d, f)) as img:
                img.convert("RGB")
        except Exception as e:
            bad += 1
            print("تالف:", os.path.join(d, f))

print(f"✅ فُحصت {total} صورة | التالف: {bad}")