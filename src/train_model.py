# الاستيرادات الأساسية
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ثوابت المشروع
IMAGE_SIZE = 224        # مقاس الصورة الذي تتوقعه القاعدة المدرَّبة
LEARNING_RATE = 0.001   # معدل التعلم: حجم خطوة تصحيح الأوزان
DROPOUT_RATE = 0.3      # معدل الإسقاط: نسبة التعطيل العشوائي لمنع الحفظ
BATCH_SIZE = 32         # حجم الدفعة: صور كل مجموعة قبل تصحيح
EPOCHS = 10             # عدد الحقب الأقصى للتدريب
TRAIN_DIR = "data/train"  # مجلد صور التدريب
VAL_DIR = "data/val"      # مجلد صور التحقق
TEST_DIR = "data/test"    # مجلد صور الاختبار
MODEL_PATH = "models/best_model.keras"  # مسار حفظ أفضل نموذج

# دالة بناء النوذج تغلف البناء وتستدعى من داخل train_model
def build_model():
    """
    بناء نموذج MobileNetV2 برأس تصنيف ثنائي (سليم / تسوس)

    المخرجات (Returns):
        Model: نموذج كيراس مجمّع وجاهز للتدريب
    """
    # استدعاء القاعدة mobilenetv2 المدرَّبة مسبقاً بدون رأسها الأصلي
    base_model = MobileNetV2(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    # تجميد القاعدة: حماية خبرتها من بياناتنا الصغيرة
    base_model.trainable = False
    # إعلان شكل الدخول — بداية الخط الذي سيتتبعه Model
    inputs = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(DROPOUT_RATE)(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def preprocess_image(image_path):
    """
    تحميل صورة وتجهيزها للنموذج بدون أي اعتماد على OpenCV

    المعطيات (Args):
        image_path (str): مسار ملف الصورة

    المخرجات (Returns):
        tf.Tensor: مصفوفة بشكل (1, 224, 224, 3) جاهزة للإطعام للنموذج
    """
    img = tf.keras.utils.load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    arr = tf.keras.utils.img_to_array(img)
    arr = tf.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr


def make_generators():
    """
    إنشاء حزامَي التدريب (بزيادة) والتحقق (بدون زيادة)

    المخرجات (Returns):
        tuple: (train_loader، val_loader)
    """
    train_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
    )
    val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_loader = train_gen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="binary",
    )
    val_loader = val_gen.flow_from_directory(
        VAL_DIR,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="binary",
    )
    return train_loader, val_loader

# دالة حساب أوزان الفئات لمعالجة عدم التوازن بين تسوس وسليم
def compute_class_weight(train_loader):
    """
    حساب أوزان الفئات لمعالجة عدم التوازن بين تسوس وسليم

    المعطيات (Args):
        train_loader: حزام التدريب لقراءة أعداد الفئات

    المخرجات (Returns):
        dict: قاموس {رقم الفئة: وزنها}
    """
    labels = list(train_loader.classes)   # خاصية: رقم تصنيف كل صورة
    n_caries = labels.count(0)
    n_healthy = labels.count(1)
    total = n_caries + n_healthy
    return {0: total / (2 * n_caries), 1: total / (2 * n_healthy)}

# دالة التوقف المبكر وحفظ أفضل نموذج
def make_callbacks():
    """
    إنشاء حارسي التدريب: حفظ الأفضل والإيقاف المبكر

    المخرجات (Returns):
        list: قائمة الحراس لتمريرها إلى fit
    """
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        # بعد كل حقبة، إن كان val_loss 
        # أفضل من كل ما سبق → يكتب الملف 
        # models/best_model.keras
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", # تحديد اي رقم يراقب
        patience=3, #ثلاث حقب بدون تحسن → إيقاف التدريب
        restore_best_weights=True, #أعد افضل وزن للنموذج بعد التوقف
    )
    return [checkpoint, early_stop]

# قلب العملية: تدريب النموذج على صور التدريب والتحقق بعد كل حقبة
def train_model():
    """
    تدريب الرأس الجديد على صور التدريب مع التحقق بعد كل حقبة

    المخرجات (Returns):
        tuple: (النموذج المدرَّب، سجل التدريب history)
    """
    model = build_model()
    train_loader, val_loader = make_generators()
    class_weight = compute_class_weight(train_loader)
    print("Class weights:", class_weight)
    print("أوزان الفئات:", class_weight)
    history = model.fit(
        train_loader,
        epochs=EPOCHS,
        validation_data=val_loader,
        class_weight=class_weight,
        callbacks=make_callbacks(),
    )
    return model, history

# شرط تشغيل الملف مباشرة: تدريب النموذج وحفظه، ثم تقييمه على بيانات الاختبار
if __name__ == "__main__":
    # التدريب الكامل / Full training
    model, history = train_model()

    # الامتحان النهائي مرة واحدة على test / Final exam on test once
    test_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_loader = test_gen.flow_from_directory(
        TEST_DIR,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False,
    )
    test_loss, test_acc = model.evaluate(test_loader)
    print("Final test loss:", test_loss)
    print("خسارة الاختبار النهائية:", test_loss)
    print("Final test accuracy:", test_acc)
    print("دقة الاختبار النهائية:", test_acc)
    print("Best model saved at:", MODEL_PATH)
    print("تم حفظ أفضل نموذج في:", MODEL_PATH)
































# # الاستيرادات الأساسية
# import tensorflow as tf
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
# from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout
# from tensorflow.keras.models import Model

# # ثوابت المشروع
# IMAGE_SIZE = 224        # مقاس الصورة الذي تتوقعه القاعدة المدرَّبة
# LEARNING_RATE = 0.001   # معدل التعلم: حجم خطوة تصحيح الأوزان
# DROPOUT_RATE = 0.3      # معدل الإسقاط: نسبة التعطيل العشوائي لمنع الحفظ


# def build_model():
#     """
#     بناء نموذج MobileNetV2 برأس تصنيف ثنائي (سليم / تسوس)

#     المخرجات (Returns):
#         Model: نموذج كيراس مجمّع وجاهز للتدريب
#     """
#     # تحميل القاعدة المدرَّبة مسبقاً بدون رأسها الأصلي
#     base_model = MobileNetV2(
#         input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
#         include_top=False,      # خلع رأس التصنيف الأصلي (1000 فئة)
#         weights="imagenet",     # تحميل الأوزان المتعلمة من ImageNet
#     )

#     # تجميد القاعدة: حماية خبرتها من بياناتنا الصغيرة
#     base_model.trainable = False

#     # بناء الرأس الجديد
#     inputs = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
#     # training=False: تثبيت طبقات التطبيع في القاعدة على وضع الاستدلال
#     x = base_model(inputs, training=False)
#     # ضغط خريطة الميزات إلى متجه ملخص ثابت الطول
#     x = GlobalAveragePooling2D()(x)
#     # طبقة كثيفة تتعلم أنماط التسوس
#     x = Dense(128, activation="relu")(x)
#     # إسقاط عشوائي لمنع الإفراط في التلائم
#     x = Dropout(DROPOUT_RATE)(x)
#     # طبقة الخرج: احتمال تسوس بين 0 و1
#     outputs = Dense(1, activation="sigmoid")(x)

#     # تجميع النموذج وتجهيزه للتدريب
#     model = Model(inputs, outputs)
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
#         loss="binary_crossentropy",   # دالة خسارة التصنيف الثنائي
#         metrics=["accuracy"],
#     )
#     return model


# def preprocess_image(image_path):
#     """
#     تحميل صورة وتجهيزها للنموذج بدون أي اعتماد على OpenCV

#     المعطيات (Args):
#         image_path (str): مسار ملف الصورة

#     المخرجات (Returns):
#         tf.Tensor: مصفوفة بشكل (1, 224, 224, 3) جاهزة للإطعام للنموذج
#     """
#     # تحميل الصورة وإعادة تحجيمها للمقاس المطلوب بخطوة واحدة (تستخدم PIL)
#     img = tf.keras.utils.load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
#     # تحويل كائن الصورة إلى مصفوفة أرقام (224, 224, 3)
#     arr = tf.keras.utils.img_to_array(img)
#     # إضافة بُعد الدفعة: (224,224,3) → (1,224,224,3)
#     arr = tf.expand_dims(arr, axis=0)
#     # قياس القيم إلى المدى الذي تدرّبت عليه القاعدة [-1, 1]
#     arr = preprocess_input(arr)
#     return arr


# if __name__ == "__main__":
#     # بناء النموذج وطباعة تقرير الفحص الفني للتحقق
#     model = build_model()
#     model.summary()