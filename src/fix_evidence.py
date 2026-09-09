# أدلة إصلاح قلب الترجمة: مقارنة قبل/بعد على مجموعة الاختبار
import tensorflow as tf
from train_model import MODEL_PATH, IMAGE_SIZE, BATCH_SIZE, TEST_DIR
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

CLASS_NAMES = ["caries", "healthy"]

model = tf.keras.models.load_model(MODEL_PATH)
gen = ImageDataGenerator(preprocessing_function=preprocess_input)
loader = gen.flow_from_directory(
    TEST_DIR, target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE, class_mode="binary", shuffle=False,
)

probs = model.predict(loader, verbose=0).flatten()
truth = loader.classes
filenames = loader.filenames

idx_after = (probs > 0.5).astype(int)   # الترجمة المصححة
idx_before = 1 - idx_after              # الترجمة المعطوبة (قلب دائم)


def report(tag, pred_idx):
    tp = int(((truth == 0) & (pred_idx == 0)).sum())
    fn = int(((truth == 0) & (pred_idx == 1)).sum())
    fp = int(((truth == 1) & (pred_idx == 0)).sum())
    tn = int(((truth == 1) & (pred_idx == 1)).sum())
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    print(f"--- {tag} ---")
    print(f"  true caries : pred caries={tp}  pred healthy={fn}")
    print(f"  true healthy: pred caries={fp}  pred healthy={tn}")
    print(f"  Caries Recall: {recall:.4f}   Accuracy: {(tp + tn) / len(truth):.4f}")
    print()


report("BEFORE FIX", idx_before)
report("AFTER FIX", idx_after)

print("Caries images wrong BEFORE fix and their NEW label:")
for name, t, b, a in zip(filenames, truth, idx_before, idx_after):
    if t == 0 and b != t:
        print(f"  {name}: before={CLASS_NAMES[b]}  after={CLASS_NAMES[a]}")
print()
print("Caries images STILL wrong AFTER fix (natural model errors):")
for name, t, a in zip(filenames, truth, idx_after):
    if t == 0 and a != t:
        print(f"  {name}: after={CLASS_NAMES[a]}")