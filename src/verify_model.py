# التحقق المستقل: تحميل النموذج المحفوظ وإعادة تقييمه من الصفر
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

MODEL_PATH = "models/best_model.keras"
TEST_DIR = "data/test"

# تحميل النموذج من القرص (لا من الذاكرة) — دليل أن الملف صالح
model = tf.keras.models.load_model(MODEL_PATH)

test_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_loader = test_gen.flow_from_directory(
    TEST_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    shuffle=False,
)

# إعادة الحساب من الصفر
loss, acc = model.evaluate(test_loader)
print("Reproduced test accuracy:", acc)

# العد صورة صورة: كم صورة أصابها النموذج؟
probs = model.predict(test_loader, verbose=0)
preds = (probs > 0.5).astype(int).flatten()   # احتمال > 0.5 يعني تسوس
truth = test_loader.classes                    # الحقيقة من المجلدات
correct = int((preds == truth).sum())
print("Correct images:", correct, "/", len(truth))