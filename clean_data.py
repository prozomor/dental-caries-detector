import os, shutil
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor
from imagededup.methods import PHash

RAW  = "data/raw"
REJ  = "data/rejected"
EXTS = ('.jpg', '.jpeg', '.png')

# ---------- المرحلة 1: حذف المكررات ----------
ph = PHash()
for cls in os.listdir(RAW):
    src = os.path.join(RAW, cls)
    if not os.path.isdir(src): continue
    try:
        dups = ph.find_duplicates_to_remove(image_dir=src)
        for f in dups:
            os.remove(os.path.join(src, f))
        print(f"[تكرار] {cls}: حُذف {len(dups)}")
    except Exception as e:
        print(f"[تكرار] {cls}: تخطي ({e})")

# ---------- المرحلة 2: فرز CLIP ----------
MODEL = "openai/clip-vit-base-patch32"
print("تحميل CLIP (من الكاش هذه المرة)...")
model = CLIPModel.from_pretrained(MODEL)
processor = CLIPProcessor.from_pretrained(MODEL)
model.eval()

TEXTS = [
    "a close-up clinical photograph of real human teeth inside an open mouth",
    "a dental X-ray radiograph",
    "a 3D rendered computer illustration of teeth",
    "a plastic dental model or denture on a table",
    "a slide or poster with written text about teeth",
    "a photo of a whole person face, not a mouth close-up",
]

kept = rej = 0
for cls in os.listdir(RAW):
    src = os.path.join(RAW, cls)
    if not os.path.isdir(src): continue
    for f in sorted(os.listdir(src)):
        if not f.lower().endswith(EXTS): continue
        p = os.path.join(src, f)
        try:
            img = Image.open(p).convert("RGB")
            inputs = processor(text=TEXTS, images=img,
                               return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                logits = model(**inputs).logits_per_image[0]
            ok = int(logits.argmax().item()) == 0
        except Exception:
            ok = False
        if ok:
            kept += 1
        else:
            dst_dir = os.path.join(REJ, cls)
            os.makedirs(dst_dir, exist_ok=True)
            shutil.move(p, os.path.join(dst_dir, f))
            rej += 1
    print(f"[CLIP] {cls}: تم الفحص")

print(f"✅ احتُفظ: {kept} | عُزل: {rej}")