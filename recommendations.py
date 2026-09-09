# --- نظام التوصيات الذكي (إكمال B4) ---
# دالة نقية: تدخلها نتيجة التنبؤ، وتخرج قائمة نصائح تناسب القرار ومنطقة الثقة
def build_recommendations(prediction):
    cls = prediction["class"]
    zone = prediction["zone"]
    if zone == "LOW":
        return ["النموذج امتنع عن القرار: يلزم فحص سريري عند طبيب الأسنان",
                "لا تعتمد على هذه النتيجة في أي إجراء"]
    recs = []
    if cls == "caries":
        recs += ["حجز موعد عند طبيب الأسنان لتقييم المعالجة",
                 "تقليل السكريات وتنظيف مرتين يومياً بمعجون فلورايد"]
    else:
        recs += ["المحافظة على التنظيف مرتين يومياً والخيط",
                 "فحص دوري كل 6 أشهر"]
    if zone == "MEDIUM":
        recs.append("الثقة متوسطة: يُستحسن إعادة التصوير بإضاءة أفضل أو التأكيد")
    return recs


if __name__ == "__main__":
    # اختبار بثلاث حالات افتراضية — لا يحتاج tensorflow
    print(build_recommendations({"class": "caries", "zone": "HIGH"}))
    print(build_recommendations({"class": "healthy", "zone": "MEDIUM"}))
    print(build_recommendations({"class": "healthy", "zone": "LOW"}))