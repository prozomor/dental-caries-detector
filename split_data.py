# ============================================
# سكربت تقسيم البيانات (Data Splitting)
# الهدف: تقسيم الصور إلى train / val / test
# ============================================

# --- الاستيرادات ---
import os        # لقراءة المجلدات والملفات
import shutil    # لنسخ الملفات
import random    # لخلط الصور

# --- الإعدادات (مفاتيح الضبط) ---
SEED = 42                        # رقم ثابت لتكرار نفس الخلط على الجهازين
BASE_DIR = "data"                # المجلد الأساسي
RAW_DIR = os.path.join(BASE_DIR, "raw")   # مسار الصور الأصلية

TRAIN_RATIO = 0.70               # 70% تدريب
VAL_RATIO   = 0.15               # 15% ضبط
TEST_RATIO  = 0.15               # 15% امتحان نهائي

CLASSES = ["caries", "healthy"]  # الفئتان

# --- دالة تقسيم فئة واحدة ---
def split_class(class_name):
    class_dir = os.path.join(RAW_DIR, class_name)      # مسار مجلد الفئة
    images = sorted(os.listdir(class_dir))             # قراءة الأسماء وترتيبها
    random.seed(SEED)                                  # تثبيت "رقم الحظ"
    random.shuffle(images)                             # خلط قابل للتكرار
    total_images = len(images)                         # العدد الكلي
    train_count = int(total_images * TRAIN_RATIO)      # عدد التدريب
    val_count = int(total_images * VAL_RATIO)          # عدد التحقق
    train_images = images[:train_count]                # شريحة التدريب
    val_images = images[train_count:train_count + val_count]  # شريحة التحقق
    test_images = images[train_count + val_count:]     # الباقي للاختبار
    return train_images, val_images, test_images

# --- دالة النسخ ---
def copy_images(image_list, class_name, split_name):
    dest_dir = os.path.join(BASE_DIR, split_name, class_name)  # المجلد الوجهة
    os.makedirs(dest_dir, exist_ok=True)               # إنشاؤه إن لم يوجد
    for image_name in image_list:                      # لكل اسم صورة
        src = os.path.join(RAW_DIR, class_name, image_name)   # المصدر
        dst = os.path.join(dest_dir, image_name)              # الوجهة
        shutil.copy2(src, dst)                         # نسخ مع البيانات الوصفية

# --- المحرك الرئيسي (يعمل فقط عند التشغيل المباشر) ---
if __name__ == "__main__":
    for class_name in CLASSES:
        train_imgs, val_imgs, test_imgs = split_class(class_name)
        copy_images(train_imgs, class_name, "train")
        copy_images(val_imgs, class_name, "val")
        copy_images(test_imgs, class_name, "test")
        print(f"{class_name}: train={len(train_imgs)}, val={len(val_imgs)}, test={len(test_imgs)}")
    print("تم التقسيم بنجاح!")