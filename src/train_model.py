# الاستيرادات الأساسية
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

# ثوابت المشروع
IMAGE_SIZE = 224        # مقاس الصورة الذي تتوقعه القاعدة المدرَّبة
LEARNING_RATE = 0.001   # معدل التعلم: حجم خطوة تصحيح الأوزان
DROPOUT_RATE = 0.3      # معدل الإسقاط: نسبة التعطيل العشوائي لمنع الحفظ


def build_model():
    """
    بناء نموذج MobileNetV2 برأس تصنيف ثنائي (سليم / تسوس)

    المخرجات (Returns):
        Model: نموذج كيراس مجمّع وجاهز للتدريب
    """
    # تحميل القاعدة المدرَّبة مسبقاً بدون رأسها الأصلي
    base_model = MobileNetV2(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        include_top=False,      # خلع رأس التصنيف الأصلي (1000 فئة)
        weights="imagenet",     # تحميل الأوزان المتعلمة من ImageNet
    )

    # تجميد القاعدة: حماية خبرتها من بياناتنا الصغيرة
    base_model.trainable = False

    # بناء الرأس الجديد
    inputs = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    # training=False: تثبيت طبقات التطبيع في القاعدة على وضع الاستدلال
    x = base_model(inputs, training=False)
    # ضغط خريطة الميزات إلى متجه ملخص ثابت الطول
    x = GlobalAveragePooling2D()(x)
    # طبقة كثيفة تتعلم أنماط التسوس
    x = Dense(128, activation="relu")(x)
    # إسقاط عشوائي لمنع الإفراط في التلائم
    x = Dropout(DROPOUT_RATE)(x)
    # طبقة الخرج: احتمال تسوس بين 0 و1
    outputs = Dense(1, activation="sigmoid")(x)

    # تجميع النموذج وتجهيزه للتدريب
    model = Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss="binary_crossentropy",   # دالة خسارة التصنيف الثنائي
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
    # تحميل الصورة وإعادة تحجيمها للمقاس المطلوب بخطوة واحدة (تستخدم PIL)
    img = tf.keras.utils.load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    # تحويل كائن الصورة إلى مصفوفة أرقام (224, 224, 3)
    arr = tf.keras.utils.img_to_array(img)
    # إضافة بُعد الدفعة: (224,224,3) → (1,224,224,3)
    arr = tf.expand_dims(arr, axis=0)
    # قياس القيم إلى المدى الذي تدرّبت عليه القاعدة [-1, 1]
    arr = preprocess_input(arr)
    return arr


if __name__ == "__main__":
    # بناء النموذج وطباعة تقرير الفحص الفني للتحقق
    model = build_model()
    model.summary()