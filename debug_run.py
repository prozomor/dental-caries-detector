# debug_run.py — يشغّل المسارين منفصلين ليكشف الجاني باسمه
import traceback
from app import process_cv_pipeline, process_cnn_pipeline

PATH = r"data/test/caries/127.jpg"   # صورة معروفة من تقسيمنا

print("=== CV PATH ===")
try:
    panels, indicators = process_cv_pipeline(PATH)
    print("CV OK")
    print(indicators)
except Exception:
    traceback.print_exc()

print("=== CNN PATH ===")
try:
    text, heat = process_cnn_pipeline(PATH)
    print("CNN OK")
    print(text)
except Exception:
    traceback.print_exc()