# استيراد أدوات القرار من ملف التدريب (مصدر واحد بلا تكرار)
from train_model import MODEL_PATH, preprocess_image
import tensorflow as tf

# أسماء الفئات بنفس ترتيب ترقيم المولّد الأبجدي: caries=0 , healthy=1
CLASS_NAMES = ["caries", "healthy"]

# عتبات مناطق الثقة (بدايات - تُضبط في J3 على صور حقيقية)
HIGH_THRESHOLD = 0.85    # فوقها: منطقة الثقة العالية
MEDIUM_THRESHOLD = 0.65  # فوقها: متوسطة، وتحتها: منخفضة (لا قرار)


def predict_image(image_path, model=None):
    """
    توقع تصنيف صورة واحدة وإرجاع القرار مع نسبة الثقة

    المعطيات (Args):
        image_path (str): مسار ملف الصورة
        model: نموذج محمّل مسبقاً (اختياري - إن لم يُمرر يُحمّل من القرص)

    المخرجات (Returns):
        dict: {"class": "caries"/"healthy", "confidence": نسبة الثقة}
    """
    # إن لم يُمرر نموذج، حمّله من القرص (الملف الثنائي)
    if model is None:
        model = tf.keras.models.load_model(MODEL_PATH)
    # تجهيز الصورة بلغة الطبيب (دالتنا من B1)
    arr = preprocess_image(image_path)
    # القاضي يخرج احتمال الفئة رقم 1 أبجدياً = احتمال "healthy"
    prob_healthy = float(model.predict(arr, verbose=0)[0][0])
    # القرار: فوق التعادل = الفئة 1 (healthy)، تحته = الفئة 0 (caries)
    idx = 1 if prob_healthy > 0.5 else 0
    # ترجمة الفهرس إلى اسم عبر القائمة الثابتة (لا افتراضات)
    label = CLASS_NAMES[idx]
    # الثقة: قوة الاقتناع بالقرار المختار (دائماً بين 0.5 و1)
    confidence = prob_healthy if idx == 1 else 1.0 - prob_healthy
    return {"class": label, "confidence": confidence}


def assess_confidence(confidence):
    """
    تصنيف الثقة إلى إحدى ثلاث مناطق وإرجاع سلوك الأداة المناسب

    المعطيات (Args):
        confidence (float): نسبة الثقة بين 0.5 و1

    المخرجات (Returns):
        dict: {"zone": "HIGH"/"MEDIUM"/"LOW", "message": رسالة السلوك}
    """
    # منطقة عالية: قرار واضح
    if confidence >= HIGH_THRESHOLD:
        zone, message = "HIGH", "قرار واضح - اتبع التوصية"
    # منطقة متوسطة: قرار بحذر
    elif confidence >= MEDIUM_THRESHOLD:
        zone, message = "MEDIUM", "قرار بحذر - يُستحسن التأكيد"
    # منطقة منخفضة: الأداة تمتنع عن القرار (أخلاقياً: لا تخمين في صحة إنسان)
    else:
        zone, message = "LOW", "النموذج لا يقرر - يلزم فحص سريري"
    return {"zone": zone, "message": message}


def full_prediction(image_path, model=None):
    """
    توقع كامل بنداء واحد: تصنيف + ثقة + منطقة + رسالة

    المعطيات (Args):
        image_path (str): مسار ملف الصورة
        model: نموذج محمّل مسبقاً (اختياري)

    المخرجات (Returns):
        dict: نتيجة دمج predict_image مع assess_confidence
    """
    result = predict_image(image_path, model)
    result.update(assess_confidence(result["confidence"]))
    return result


if __name__ == "__main__":
    # اختبار على صور معروفة التصنيف من مجلدات الاختبار
    print("caries 1 :", full_prediction("data/test/caries/80.jpg"))
    print("caries 4 :", full_prediction("data/test/caries/515.jpg"))
    print("healthy 1:", full_prediction("data/test/healthy/267.jpg"))


# if __name__ == "__main__":
#     print("caries 1 :", full_prediction("data/test/caries/127.jpg"))
#     print("caries 4 :", full_prediction("data/test/caries/515.jpg"))
#     print("healthy 1:", full_prediction("data/test/healthy/92.jpg"))