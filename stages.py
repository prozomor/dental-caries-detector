# ============================================
# مراحل معالجة الصور (المسار الثاني — OpenCV)
# ============================================

import cv2          # مكتبة OpenCV لمعالجة الصور
import numpy as np  # مكتبة المصفوفات والأرقام

# --- تحميل الصورة بأمان (يدعم المسارات العربية) ---
def load_image(image_path):
    data = np.fromfile(image_path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"تعذر تحميل الصورة: {image_path}")
    return image

# --- حفظ الصورة بأمان (يدعم المسارات العربية) ---
def save_image(image, output_path):
    _, data = cv2.imencode(".png", image)
    data.tofile(output_path)

# --- المرحلة 1: التحويل إلى رمادي ---
def to_grayscale(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

# --- المرحلة 2: تحسين التباين CLAHE ---
def apply_clahe(gray_image, clip_limit=2.0, tile_size=8):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    enhanced = clahe.apply(gray_image)
    return enhanced

# --- المرحلة 3: إزالة الضجيج (التنعيم) ---
def reduce_noise(gray_image, kernel_size=5):
    smoothed = cv2.GaussianBlur(gray_image, (kernel_size, kernel_size), 0)
    return smoothed

# --- المرحلة 4: كشف الحواف (Canny) ---
def detect_edges(smoothed_image, low_threshold=50, high_threshold=150):
    edges = cv2.Canny(smoothed_image, low_threshold, high_threshold)
    return edges

# --- المرحلة 5: التقسيم بعتبة أوتسو ---
def segment_otsu(gray_image):
    threshold_value, binary = cv2.threshold(
        gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return threshold_value, binary

# --- المرحلة 6: الكنتورات (مربعات حول المناطق) ---
def draw_contours(binary_mask, color_image, min_area=500):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = color_image.copy()
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return result

# --- المرحلة 7: كاشف المناطق التسوسية اللونية (HSV) ---
def detect_caries_candidates(color_image, min_area=200):
    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    # نطاق أضق: بني داكن فقط (يستبعد وردي اللثة الفاتح والرمادي المعدني)
    lower_brown = np.array([8, 60, 20])
    upper_brown = np.array([25, 255, 150])
    mask = cv2.inRange(hsv, lower_brown, upper_brown)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = color_image.copy()
    h_img, w_img = color_image.shape[:2]
    max_area = 0.2 * h_img * w_img   # نتجاهل أي منطقة ضخمة (ليست تسوساً)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:   # لا صغيرة جداً ولا ضخمة
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return result
# --- المرحلة 9: تحليل البلاك (المهمة A3) ---
# --- الدالة 1 (محسّنة): عزل منطقة السن كقناع مملوء عبر مصفاة لونية ---
def extract_tooth_mask(color_image):
    # نحوّل لنظام HSV لنفصل "نوع اللون" عن "الإضاءة"
    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    # نافذة الأسنان: مصفر/أبيض (H 10-45)، تشبع متوسط، إضاءة عالية (V>=120)
    lower_tooth = np.array([10, 10, 120])
    upper_tooth = np.array([45, 180, 255])
    mask = cv2.inRange(hsv, lower_tooth, upper_tooth)
    # ننظف ذرات الغبار قبل اختيار الجزر
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # نمشي حول الجزر البيضاء المتبقية (الحدود الخارجية فقط)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # دفاع: إن لم توجد أي جزيرة، نُرجع قناعاً فارغاً بدل أن ينهار البرنامج
    if len(contours) == 0:
        return np.zeros_like(mask)
    # نختار أكبر جزيرة (افتراض: أكبر منطقة فاتحة مصفرة = منطقة الأسنان)
    largest = max(contours, key=cv2.contourArea)
    # لوحة سوداء فارغة بنفس الأبعاد
    tooth_mask = np.zeros_like(mask)
    # نرسم أكبر جزيرة مملوءة: تغلق ثقوب التسوس والحشوة داخل السن
    cv2.drawContours(tooth_mask, [largest], -1, 255, thickness=cv2.FILLED)
    return tooth_mask


# --- الدالة 2: حساب مساحة السن (عدد البكسلات البيضاء) ---
def calculate_tooth_area(tooth_mask):
    # المساحة = عدد البكسلات غير الصفرية (البيضاء) في القناع
    return np.count_nonzero(tooth_mask)

# --- المرحلة 9: قناع البلاك داخل منطقة السن ---
def detect_plaque_mask(color_image, tooth_mask):
    # نحوّل لنظام HSV لنفصل "نوع اللون" عن "الإضاءة"
    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    # نافذة البلاك: أصفر/مصفر مطفي (ليس وردياً وليس داكناً)
    lower_plaque = np.array([15, 40, 140])
    upper_plaque = np.array([40, 150, 255])
    plaque_candidate = cv2.inRange(hsv, lower_plaque, upper_plaque)
    # نحصر البحث داخل منطقة السن فقط (تقاطع القناعين)
    plaque_mask = cv2.bitwise_and(plaque_candidate, tooth_mask)
    return plaque_mask


# --- المرحلة 10: مؤشر البلاك (نسبة مساحة البلاك من مساحة السن) ---
def calculate_plaque_index(plaque_mask, tooth_mask):
    # نعدّ بكسلات البلاك (البيضاء في قناع البلاك)
    plaque_area = np.count_nonzero(plaque_mask)
    # نعدّ بكسلات السن (البيضاء في قناع السن)
    tooth_area = np.count_nonzero(tooth_mask)
    # دفاع: إن كانت مساحة السن صفراً (لا جزيرة)، نرجع صفراً بدل الانهيار
    if tooth_area == 0:
        return 0.0
    # النسبة المئوية = الجزء ÷ الكل × 100
    plaque_index = (plaque_area / tooth_area) * 100
    return plaque_index

# --- المرحلة 11: كشف الاصفرار بالهيستوغرام (المهمة A4) ---
def detect_yellowing(color_image, tooth_mask):
    # 1) نحوّل لنظام HSV لنتعامل مع "نوع اللون" منفصلاً
    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    # 2) قناع "المصوّتين" = البكسلات ذات تشبع كافٍ (لها لون حقيقي)
    saturated_mask = cv2.inRange(hsv, np.array([0, 40, 0]), np.array([179, 255, 255]))
    # 3) حصر التصويت داخل منطقة السن فقط (تقاطع)
    voting_mask = cv2.bitwise_and(saturated_mask, tooth_mask)
    # 4) بناء الهيستوغرام: 180 خانة (واحدة لكل درجة لون) على قناع التصويت
    hist = cv2.calcHist([hsv], [0], voting_mask, [180], [0, 180])
    # 5) جمع الخانات 15-35 = بكسلات النطاق الأصفر
    yellow_pixels = int(hist[15:36].sum())
    # 6) مساحة السن الكاملة (المقام) — تشمل البياض المشروع
    tooth_area = np.count_nonzero(tooth_mask)
    # 7) حارس: إن كانت مساحة السن صفراً، نرجع صفر بدل الانهيار
    if tooth_area == 0:
        return 0.0, "غير محسوب"
    # 8) النسبة المئوية + التقدير البسيط
    ratio = (yellow_pixels / tooth_area) * 100
    if ratio < 10:
        label = "مائل للبياض"
    elif ratio <= 30:
        label = "اصفرار خفيف"
    else:
        label = "اصفرار واضح"
    return round(ratio, 2), label

# --- اختبار شامل (يعمل فقط عند التشغيل المباشر) ---
if __name__ == "__main__":
    # 1) تحميل الصورة
    image = load_image("test_image.jpg")
    print("شكل الصورة الملونة:", image.shape)
    # 2) تحويل لرمادي
    gray = to_grayscale(image)
    # 3) تحسين التباين
    enhanced = apply_clahe(gray)
    # 4) تنعيم
    smoothed = reduce_noise(enhanced)
    # 5) حواف
    edges = detect_edges(smoothed)
    # 6) تقسيم أوتسو
    otsu_val, binary = segment_otsu(smoothed)
    print("العتبة المثلى التي وجدها Otsu:", otsu_val)
    # 7) كنتورات (محدد مناطق)
    result = draw_contours(binary, image)
    # 8) مؤشر التصبغات (مربعات حمراء)
    candidates = detect_caries_candidates(image)
    # 9) قناع السن المصمت
    tooth_mask = extract_tooth_mask(image)
    # 10) مساحة السن (المقام)
    tooth_area = calculate_tooth_area(tooth_mask)
    print("مساحة منطقة السن (بكسل):", tooth_area)
    # 11) قناع البلاك داخل السن
    plaque_mask = detect_plaque_mask(image, tooth_mask)
    # 12) مؤشر البلاك (النسبة المئوية)
    plaque_index = calculate_plaque_index(plaque_mask, tooth_mask)
    print("مؤشر البلاك (%):", round(plaque_index, 2))
    # 13) كشف الاصفرار بالهيستوغرام
    yellow_ratio, yellow_label = detect_yellowing(image, tooth_mask)
    print(f"الاصفرار: {yellow_label} ({yellow_ratio}%)")
    # حفظ كل اللوحات
    save_image(image, "01_original.png")
    save_image(gray, "02_gray.png")
    save_image(enhanced, "03_clahe.png")
    save_image(smoothed, "04_smooth.png")
    save_image(edges, "05_edges.png")
    save_image(binary, "06_otsu.png")
    save_image(result, "07_contours.png")
    save_image(candidates, "08_caries_candidates.png")
    save_image(tooth_mask, "09_tooth_mask.png")
    save_image(plaque_mask, "10_plaque_mask.png")
    print("تم حفظ كل الصور بنجاح!")