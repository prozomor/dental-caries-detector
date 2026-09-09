import os, sys
# جسر المسارات: ملفات الجذر ترى ملفات src/ (ضروري لمسار CNN)
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import gradio as gr
import logging
import traceback
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from stages import (load_image, to_grayscale, apply_clahe, reduce_noise,
                        detect_edges, segment_otsu, draw_contours,
                        detect_caries_candidates, extract_tooth_mask,
                        calculate_tooth_area, detect_plaque_mask,
                        calculate_plaque_index, detect_yellowing)
except ImportError as e:
    logger.error(f"خطأ في استيراد دوال المعالجة الأساسية: {e}")
    raise

CNN_AVAILABLE = False
MODEL = None
try:
    import tensorflow as tf
    from train_model import MODEL_PATH
    from confidence_system import full_prediction
    from recommendations import build_recommendations
    from gradcam import make_gradcam
    MODEL = tf.keras.models.load_model(MODEL_PATH)
    CNN_AVAILABLE = True
except Exception:
    CNN_AVAILABLE = False

ABOUT = """### 🔍 عن نظام الفحص الذكي لصحة الأسنان
* **مسار OpenCV:** خط أنابيب من 14 دالة (10 لوحات مرئية + 4 مؤشرات رقمية).
* **مسار CNN:** MobileNetV2 مع نظام ثقة ثلاثي المناطق يمتنع أخلاقياً عن القرار حين يتردد.
* **Grad-CAM:** خريطة توضح أين نظر النموذج قبل قراره.
* **ملاحظة طبية:** النظام أداة مساعدة (Decision Support) وليس بديلاً عن طبيب الأسنان.
"""

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;800&display=swap');

/* الصفحة وكل المكوّنات: كحلي موحّد */
.gradio-container { background:#0b1220 !important; font-family:'Tajawal',sans-serif !important; direction:rtl; }
.gradio-container .block, .gradio-container .gr-block,
.gradio-container .image-container, .gradio-container .upload-container,
.gradio-container .gallery, .gradio-container .tabs, .gradio-container .tab-nav,
.gradio-container .panel, .gradio-container .container,
.gradio-container .download-container { background:#0f1a2e !important; border-color:#1e293b !important; }
.gradio-container p, .gradio-container span, .gradio-container label,
.gradio-container .block-label, .gradio-container .prose,
.gradio-container .tab-nav button { color:#e2e8f0 !important; }
.tab-nav button.selected { color:#2dd4bf !important; border-bottom:2px solid #2dd4bf !important; }
.gradio-container footer { background:#0b1220 !important; }
.gradio-container footer * { color:#64748b !important; }

.custom-header { background:linear-gradient(135deg,#0f766e 0%,#0d9488 60%,#14b8a6 100%);
    color:#fff; padding:26px 20px; border-radius:16px; text-align:center;
    box-shadow:0 8px 24px rgba(20,184,166,.25); margin-bottom:20px; }
.custom-header h1 { margin:0; font-size:2.1rem; font-weight:800; color:#fff; }
.custom-header p { margin:8px 0 0; font-size:1.05rem; opacity:.92; }

.info-card { background:#0f1a2e; border:1px solid #1e293b; border-right:6px solid #14b8a6;
    border-radius:14px; padding:18px 20px; box-shadow:0 3px 12px rgba(0,0,0,.35);
    margin-bottom:14px; color:#e2e8f0; }
.info-card h3 { margin:0 0 12px; color:#5eead4; font-weight:800; border-bottom:1px solid #1e293b; padding-bottom:10px; }
.data-table { width:100%; border-collapse:collapse; }
.data-table td { padding:9px 6px; border-bottom:1px solid #1e293b; font-size:1.05rem; color:#e2e8f0; }
.data-table td:nth-child(2) { font-weight:800; text-align:left; }
.progress-container { background:#1e293b; border-radius:999px; height:14px; overflow:hidden; margin:8px 0 4px; }
.progress-bar { height:100%; border-radius:999px; transition:width .5s ease; }
.placeholder { color:#94a3b8; text-align:center; padding:26px 10px; font-size:1.05rem; }
"""

def generate_cv_html(otsu_val, tooth_area, plaque_index, yellow_label, yellow_ratio):
    plaque_color = "#f87171" if plaque_index > 20 else "#34d399"
    return f"""
    <div class="info-card">
        <h3>📊 مؤشرات التحليل الرقمي (OpenCV)</h3>
        <table class="data-table">
            <tr><td>🔹 عتبة الفصل (Otsu)</td><td style="color:#cbd5e1;">{otsu_val}</td></tr>
            <tr><td>🔹 مساحة السن المكتشفة</td><td style="color:#cbd5e1;">{tooth_area:,} بكسل</td></tr>
            <tr><td>🔹 مؤشر البلاك والجير</td><td style="color:{plaque_color};">{round(plaque_index, 2)}%</td></tr>
            <tr><td>🔹 الاصفرار والتصبغات</td><td style="color:#fbbf24;">{yellow_label} ({yellow_ratio}%)</td></tr>
        </table>
    </div>"""

def generate_cnn_html(pred, recs):
    AR_NAMES = {"caries": "تسوس", "healthy": "سليم"}
    is_healthy = str(pred['class']).lower() == "healthy"
    display_name = AR_NAMES.get(pred['class'], pred['class'])
    main_color = "#34d399" if is_healthy else "#f87171"
    conf_percent = round(pred['confidence'] * 100, 1)
    recs_html = "".join([f"<li style='margin-bottom:6px;'>{r}</li>" for r in recs])
    return f"""
    <div class="info-card" style="border-right-color:{main_color};">
        <h3>🧠 تقرير التشخيص الذكي</h3>
        <div style="font-size:1.25rem;margin-bottom:8px;color:#e2e8f0;">
            الحالة المتوقعة: <strong style="color:{main_color};font-size:1.5rem;">{display_name}</strong>
        </div>
        <div style="color:#94a3b8;">مستوى ثقة النموذج: {conf_percent}%</div>
        <div class="progress-container">
            <div class="progress-bar" style="width:{conf_percent}%;background-color:{main_color};"></div>
        </div>
        <div style="margin-top:12px;color:#e2e8f0;">
            <strong>📍 منطقة القرار:</strong> {pred['zone']}
            <span style="color:#94a3b8;">({pred['message']})</span>
        </div>
        <div style="margin-top:12px;background:#16233a;padding:14px;border-radius:10px;">
            <strong style="color:#5eead4;">💡 التوصيات:</strong>
            <ul style="margin:8px 0 0;padding-inline-start:20px;color:#cbd5e1;">{recs_html}</ul>
        </div>
    </div>"""

PLACE_CV = "<div class='info-card'><div class='placeholder'>📊 مؤشرات OpenCV ستظهر هنا بعد التحليل</div></div>"
PLACE_CNN = "<div class='info-card'><div class='placeholder'>🧠 تقرير الذكاء الاصطناعي سيظهر هنا بعد التحليل</div></div>"

def analyze_image(image_path):
    if not image_path:
        raise gr.Error("الرجاء رفع صورة الأسنان أولاً.")
    try:
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

        panels = [image, gray, enhanced, smoothed, edges,
                  binary, result, candidates, tooth_mask, plaque_mask]
        # ألوان العرض: OpenCV يتكلم BGR وGradio يعرض RGB
        panels = [p[..., ::-1] if (p.ndim == 3 and p.shape[2] == 3) else p for p in panels]

        cv_html = generate_cv_html(otsu_val, tooth_area, plaque_index, yellow_label, yellow_ratio)

        if CNN_AVAILABLE:
            pred = full_prediction(image_path, model=MODEL)
            recs = build_recommendations(pred)
            cnn_html = generate_cnn_html(pred, recs)
            heat = (make_gradcam(image_path, model=MODEL) * 255).astype("uint8")
        else:
            cnn_html = """
            <div class="info-card" style="border-right-color:#f59e0b;background:#2a2008;">
                <h3 style="color:#fcd34d;">⚠️ الذكاء الاصطناعي غير متوفر</h3>
                <p style="color:#fde68a;">بيئة TensorFlow غير مكتملة على هذا الجهاز — النظام يعمل كأداة تحليل صوري، ونتائج النموذج موثقة في التقرير.</p>
            </div>"""
            heat = None
        return cv_html, cnn_html, panels, heat
    except Exception as e:
        logger.error(traceback.format_exc())
        raise gr.Error(f"حدث خطأ أثناء المعالجة: {str(e)}")

with gr.Blocks(css=custom_css, title="نظام الفحص الذكي لصحة الأسنان") as demo:
    gr.HTML("""
    <div class="custom-header">
        <h1>🦷 نظام الفحص الذكي لصحة الأسنان</h1>
        <p>تحليل صوري مساعد يجمع الرؤية الحاسوبية والتعلم العميق — وليس بديلاً عن طبيب الأسنان</p>
    </div>""")

    with gr.Row():
        with gr.Column(scale=4, min_width=320):
            inp_image = gr.Image(type="filepath", label="ارفع صورة الأسنان هنا")
            analyze_btn = gr.Button("⚡ بدء التحليل الشامل", variant="primary", size="lg")
        with gr.Column(scale=6):
            cnn_results_html = gr.HTML(PLACE_CNN)
            cv_indicators_html = gr.HTML(PLACE_CV)

    with gr.Tabs():
        with gr.Tab("🖼️ مراحل المعالجة (10 لوحات)"):
            cv_gallery = gr.Gallery(columns=5, height=340, object_fit="contain", show_label=False)
        with gr.Tab("🎯 خريطة النظر (Grad-CAM)"):
            gradcam_img = gr.Image(interactive=False, height=380, show_label=False)
        with gr.Tab("ℹ️ عن النظام"):
            gr.Markdown(ABOUT)

    analyze_btn.click(fn=analyze_image, inputs=inp_image,
                      outputs=[cv_indicators_html, cnn_results_html, cv_gallery, gradcam_img])

if __name__ == "__main__":
    demo.launch(debug=False)