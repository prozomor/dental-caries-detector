
# --- واجهة نظام الفحص الذكي لصحة الأسنان (النسخة الاحترافية) ---

import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import logging
import traceback
from typing import Tuple, List, Optional, Any, Dict
import gradio as gr
import numpy as np

# 1. إعداد نظام التتبع (Logging) لمعرفة ما يحدث خلف الكواليس
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# 2. استيراد دوال المعالجة (OpenCV)
try:
    from stages import (load_image, to_grayscale, apply_clahe, reduce_noise,
                        detect_edges, segment_otsu, draw_contours,
                        detect_caries_candidates, extract_tooth_mask,
                        calculate_tooth_area, detect_plaque_mask,
                        calculate_plaque_index, detect_yellowing)
except ImportError as e:
    logger.error(f"خطأ في استيراد دوال المعالجة الأساسية: {e}")
    raise

# 3. إدارة بيئة الذكاء الاصطناعي (CNN) بأمان
CNN_AVAILABLE = False
MODEL = None

try:
    logger.info("جاري فحص بيئة TensorFlow...")
    import tensorflow as tf
    from train_model import MODEL_PATH
    from confidence_system import full_prediction
    from recommendations import build_recommendations
    from gradcam import make_gradcam
    
    MODEL = tf.keras.models.load_model(MODEL_PATH)
    CNN_AVAILABLE = True
    logger.info("تم تحميل نموذج CNN بنجاح. المسار مفعل.")
except ImportError:
    logger.warning("بيئة TensorFlow غير متوفرة. سيعمل النظام بمسار OpenCV فقط.")
except Exception as e:
    logger.error(f"حدث خطأ غير متوقع أثناء تحميل النموذج: {e}")

# 4. فصل النصوص الثابتة لتسهيل التعديل (Configuration)
ABOUT_TEXT = """### 🔍 عن النظام وحدوده الصادقة
* **مسار OpenCV (حي دائماً):** 14 دالة معالجة + 4 مؤشرات مساعدة للتحليل الأولي (ليست بديلاً عن التشخيص الطبي).
* **مسار CNN (الذكاء الاصطناعي):** يعتمد على معمارية MobileNetV2 بدقة 86.67%.
* **نظام الثقة ثلاثي المناطق:** النظام مصمم بأخلاقيات طبية؛ يمتنع عن إصدار قرار حاسم حين تكون ثقة النموذج متدنية.
* **Grad-CAM (خريطة النظر):** تقنية تفسيرية (XAI) توضح المناطق التي ركز عليها النموذج لإصدار قراره.
"""

# 5. دوال المعالجة المعزولة (Modularization)
def process_cv_pipeline(image_path: str) -> Tuple[List[Any], str]:
    """تنفذ مسار الرؤية الحاسوبية الكلاسيكية وتعيد اللوحات والمؤشرات."""
    image = load_image(image_path)
    gray = to_grayscale(image)
    enhanced = apply_clahe(gray)
    smoothed = reduce_noise(enhanced)
    edges = detect_edges(smoothed)
    otsu_val, binary = segment_otsu(smoothed)
    result = draw_contours(binary, image)
    candidates = detect_caries_candidates(image)
    tooth_mask = extract_tooth_mask(image)
    tooth_area = calculate_tooth_area(tooth_mask)
    plaque_mask = detect_plaque_mask(image, tooth_mask)
    plaque_index = calculate_plaque_index(plaque_mask, tooth_mask)
    yellow_ratio, yellow_label = detect_yellowing(image, tooth_mask)

    panels = [
        image, gray, enhanced, smoothed, edges,
        binary, result, candidates, tooth_mask, plaque_mask
    ]
    
    indicators = (
        f"🔹 عتبة Otsu: {otsu_val}\n"
        f"🔹 مساحة السن: {tooth_area:,} بكسل\n"
        f"🔹 مؤشر البلاك: {round(plaque_index, 2)}%\n"
        f"🔹 حالة الاصفرار: {yellow_label} ({yellow_ratio}%)"
    )
    return panels, indicators

def process_cnn_pipeline(image_path: str) -> Tuple[str, Optional[np.ndarray]]:
    """تنفذ مسار التعلم العميق وتعيد التقرير وخريطة النظر."""
    if not CNN_AVAILABLE:
        text = ("⚠️ مسار CNN غير متاح على هذا الجهاز (بيئة TensorFlow غير مكتملة).\n"
                "النظام يعمل الآن كأداة تحليل صوري (OpenCV) فقط.\n"
                "دقة النموذج الموثقة: 86.67% مع نظام ثقة ثلاثي المناطق.")
        return text, None

    pred = full_prediction(image_path, model=MODEL)
    recs = build_recommendations(pred)
    
    cnn_text = (
        f"🧠 الفئة المتوقعة: {pred['class']}\n"
        f"📊 نسبة الثقة: {round(pred['confidence'] * 100, 1)}%\n"
        f"📍 المنطقة: {pred['zone']} - {pred['message']}\n\n"
        f"💡 التوصيات الطبية:\n" + "\n".join(f"  - {r}" for r in recs)
    )
    
    heat = (make_gradcam(image_path, model=MODEL) * 255).astype("uint8")
    return cnn_text, heat

# 6. الدالة الرئيسية المربوطة بالواجهة (Controller)
def analyze_image(image_path: str) -> Tuple[List[Any], str, str, Optional[np.ndarray]]:
    """المدير الرئيسي الذي ينسق بين مسار CV ومسار CNN مع معالجة الأخطاء."""
    if not image_path:
        raise gr.Error("الرجاء رفع صورة أولاً!")
        
    logger.info(f"بدء تحليل الصورة: {image_path}")
    
    try:
        # تنفيذ مسار OpenCV
        panels, indicators = process_cv_pipeline(image_path)
        
        # تنفيذ مسار الذكاء الاصطناعي
        cnn_text, heat = process_cnn_pipeline(image_path)
        
        logger.info("تم التحليل بنجاح.")
        return panels, indicators, cnn_text, heat
        
    except Exception as e:
        error_msg = f"حدث خطأ أثناء تحليل الصورة: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise gr.Error("فشل في تحليل الصورة. تأكد من جودة وتنسيق الملف المرفوع.")

# 7. بناء واجهة المستخدم الاحترافية (UI/UX)
# استخدام ثيم طبي مخصص (ألوان هادئة للعين)
custom_theme = gr.themes.Soft(
    primary_hue="teal",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Cairo"), "sans-serif"]
)

with gr.Blocks(theme=custom_theme, title="نظام الفحص الذكي لصحة الأسنان", css=".gradio-container {direction: rtl; text-align: right;}") as demo:
    
    # الترويسة
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("# 🦷 نظام الفحص الذكي لصحة الأسنان")
            gr.Markdown("*تحليل صوري دقيق مدعوم بالذكاء الاصطناعي والتفسير البصري*")
    
    # قسم الإدخال
    with gr.Row():
        with gr.Column(scale=1):
            inp_image = gr.Image(type="filepath", label="ارفع صورة الأسنان هنا", height=300)
            analyze_btn = gr.Button("🚀 بدء التحليل الشامل", variant="primary", size="lg")
    
    # قسم النتائج (استخدام Tabs لتنظيم المعلومات)
    with gr.Tabs():
        with gr.Tab("📋 التقرير الشامل"):
            with gr.Row():
                cv_indicators_box = gr.Textbox(label="مؤشرات التحليل الصوري (OpenCV)", lines=5, interactive=False)
                cnn_results_box = gr.Textbox(label="التشخيص والتوصيات (CNN)", lines=5, interactive=False)
                
        with gr.Tab("🖼️ مراحل المعالجة (10 لوحات)"):
            cv_gallery = gr.Gallery(columns=5, height=400, label="خطوات المعالجة خطوة بخطوة", show_label=True)
            
        with gr.Tab("🎯 خريطة النظر (Grad-CAM)"):
            with gr.Row():
                gradcam_img = gr.Image(label="أين ركز الذكاء الاصطناعي؟", interactive=False, height=400)
                
        with gr.Tab("ℹ️ معلومات النظام"):
            gr.Markdown(ABOUT_TEXT)

    # ربط الأحداث
    analyze_btn.click(
        fn=analyze_image,
        inputs=inp_image,
        outputs=[cv_gallery, cv_indicators_box, cnn_results_box, gradcam_img],
        api_name="analyze"
    )

if __name__ == "__main__":
    # تشغيل التطبيق
    logger.info("جاري إطلاق الواجهة...")
    demo.launch(debug=False, share=False)