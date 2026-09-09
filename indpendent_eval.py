# independent_eval.py — تقييم مستقل بفريق بلال
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

MODEL_PATH = "models/best_model.keras"
TEST_DIR = "data/test"

model = tf.keras.models.load_model(MODEL_PATH)

gen = ImageDataGenerator(preprocessing_function=preprocess_input)
loader = gen.flow_from_directory(
    TEST_DIR, target_size=(224, 224), batch_size=32,
    class_mode="binary", shuffle=False,
)
print("class_indices:", loader.class_indices)   # حقيقة الخريطة من أسماء المجلدات نفسها

probs = model.predict(loader, verbose=0).flatten()   # خرج sigmoid = احتمال healthy
truth = loader.classes
c = loader.class_indices["caries"]                   # رقم فئة التسوس بالاسم لا بالتخمين

for t in [0.5, 0.6, 0.7]:
    pred_caries = (probs <= t).astype(int)           # نتنبأ تسوساً كلما قلّ احتمال healthy
    true_caries = (truth == c).astype(int)
    tp = int(((pred_caries == 1) & (true_caries == 1)).sum())
    fn = int(((pred_caries == 0) & (true_caries == 1)).sum())
    fp = int(((pred_caries == 1) & (true_caries == 0)).sum())
    tn = int(((pred_caries == 0) & (true_caries == 0)).sum())
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    print(f"threshold={t}  TP={tp} FN={fn} FP={fp} TN={tn}  caries_recall={recall:.3f}")