# الوثيقة الشاملة — مرحلة التدريب كاملة (B1 + B2)

> مشروع كشف التسوس في الأشعة السنية — مسار CNN
> إعداد: أواب | مراجعة: بلال | الإصدار: 1.0
> كُتبت لقارئ يفتحها لأول مرة — لا تفترض أي معرفة مسبقة

---

## 0) أين تقف هذه الوثيقة من المشروع

المشروع يكشف التسوس في صور الأشعة ويعطي احتمالاً (سليم / تسوس).
فريقان بالتوازي: مسار OpenCV (مؤشرات بصرية مساعدة) ومسار CNN (التشخيص الأساسي) — هذه الوثيقة تشرح مسار CNN.
البيانات: 486 صورة مقسمة (مهمة A2) إلى: تدريب 340 / تحقق 71 / اختبار 75.
مرحلة التدريب خطوتان: B1 بناء هيكل النموذج، ثم B2 تدريبه والتحقق منه.

---

## 1) الجزء النظري والمنطقي

### 1.1 الحاسوب لا يرى صوراً، يرى جداول أرقام
الصورة عنده جدول أرقام: كل خلية (بكسل) تحمل شدة اللون.
صورة 224×224 ملونة = جدول (224, 224, 3) — الثالث قنوات الألوان RGB.
إذن كل "رؤية" عند الحاسوب = حساب على أرقام.

### 1.2 الشبكة العصبية: سلّم فرق يزن الأدلة
- الخلية العصبية: ميزان يستقبل أدلة، يضرب كل دليل في وزن (أهميته)، يجمع، ويُخرج رقماً. الأوزان = "براغي الضبط" التي يتعلمها النموذج
- الطبقة: فريق من هذه الموازين؛ الأول يكتشف بسائط (حواف، قتامة)، الثاني يركّب أشكالاً (حفرة، ظل)، والأخير يحكم
- الشبكة التلافيفية CNN: نافذة صغيرة تمشط الصورة فتكتشف الدليل في أي مكان — لذلك تجد التسوس يميناً أو يساراً
- التدريب: نعرض صوراً بإجاباتها؛ يخطئ؛ المحسِّن (Optimizer) يلف كل برغي لفة صغيرة باتجاه تقليل الخطأ؛ نكرر فينزل الخطأ

### 1.3 لماذا لا نبني من الصفر؟ (التعلم بالنقل Transfer Learning)
بناء السلّم من الصفر يحتاج ملايين الصور ونحن عندنا 486.
نستعير MobileNetV2: نموذج جاهز درّب على ImageNet (1.4 مليون صورة) وفرقه الأولى تجيد الحواف والأشكال.
بقرارين:
1. نخلع رأسه القديم (1000 فئة لا تعنينا) — استخدامه كمستخرج ميزات Feature Extraction
2. نجمّد براغيه Freezing: حتى لا تمحو صورنا الـ486 خبرة ملايين الصور (نسيان كارثي + حفظ)
ثم نبني رأسنا الجديد الصغير فوقه — وهو وحده من سيتدرب.

### 1.4 محطات النموذج الست وأرقامها
| المحطة | الاسم التقني | الشكل الخارج | البراغي |
|---|---|---|---|
| الباب | Input Layer | (None,224,224,3) | 0 |
| الطبيب المجمّد | Pretrained Base (MobileNetV2) | (None,7,7,1280) خريطة ميزات | 2,257,984 ملصوقة |
| الضاغط | Global Average Pooling | (None,1280) ملخص سطر | 0 |
| المتعلم | Dense 128 + ReLU | (None,128) | 163,968 حرة |
| مانع الحفظ | Dropout 0.3 | (None,128) | 0 |
| القاضي | Dense 1 + Sigmoid | (None,1) احتمال 0→1 | 129 حرة |

الأرقام الثلاثة: Trainable = 164,097 (رأسنا فقط = دليل نجاح التجميد)،
Non-trainable = 2,257,984 (الطبيب)، Total = 2,422,081.
طبقات البراغي صفر = عمليات ثابتة لا تتعلم.

### 1.5 لغة الطبيب: القياس المسبق
MobileNetV2 تدرّب على قيم مضغوطة في المدى [-1, 1] عبر دالة preprocess_input.
إطعامه صورة خام 0–255 = مخاطبته بلغة لم يتدرب عليها → نتائج خاطئة.
القاعدة: نفس المعالجة في التدريب والاستخدام.
التحميل بأداة كيراس (PIL داخلياً) — بلا OpenCV استقلالاً عن المسار الآخر.
النموذج يستقبل دفعات؛ حتى لصورة واحدة نضيف بُعداً: (1,224,224,3).

### 1.6 مفاهيم التدريب الخمسة
- الدفعة Batch (32): مجموعة صور يتعلم منها ثم يصحح
- الحقبة Epoch: مرور الـ340 كلها مرة
- الخسارة Loss (binary_crossentropy): رقم الخطأ — منحناه يجب أن ينزل
- التحقق val (71): امتحان تجريبي بعد كل حقبة بلا تعلم — قاضٍ نزيه يكشف الحفظ
- الاختبار test (75): الامتحان النهائي — يُمس مرة واحدة بعد كل شيء
- الحفظ Overfitting: تدريب ينزل وتحقق يصعد = يحفظ الصور لا المفهوم

### 1.7 عدة التدريب وأسرار قيمها
- المولّدات: أحزمة ناقلة تقرأ المجلدات دفعات (توفير ذاكرة)؛ ترقيم الفئات أبجدياً caries=0 وhealthy=1
- الزيادة Augmentation: دوران 15° وإزاحة/تقريب 10% وقلب أفقي — قيم خفيفة مقصودة حتى لا تشوه تشريح السن؛ على train فقط
- أوزان الفئات: 219 مقابل 121 → الوزن = الكلي ÷ (2×عدد الفئة) = {0: 0.776, 1: 1.405} فيصبح خطأ الأقلية أغلى
- الحارسان: EarlyStopping (قرار التوقف بعد 3 حقب بلا تحسين) / ModelCheckpoint (حفظ الأفضل على القرص) / restore_best_weights (إعادة الأفضل للذاكرة) — ثلاثة أدوار منفصلة
- fit: حلقة التدريب؛ evaluate: الامتحان النهائي

---

## 2) ملف src/train_model.py — الشرح الكامل

### 2.1 الاستيرادات (أوامر استيراد)
| السطر | ما يُدخل | نوعه | لماذا |
|---|---|---|---|
| import tensorflow as tf | المحرك بلقب tf | وحدة | كل الحسابات والتدريب |
| ...applications import MobileNetV2 | الطبيب | صنف | بناء القاعدة الجاهزة |
| ...mobilenet_v2 import preprocess_input | دالة القياس | دالة | لغة الطبيب [-1,1] |
| ...layers import Input, GlobalAveragePooling2D, Dense, Dropout | قطع الرأس | أصناف | المحطات الأربع |
| ...models import Model | خط التجميع | صنف | تغليف المحطات بكائن |
| ...preprocessing.image import ImageDataGenerator | أمين المكتبة | صنف | قراءة المجلدات والزيادة |

### 2.2 الثوابت (متغيرات لا تُغيَّر — مصدر واحد للقيمة)
| الثابت | القيمة | لماذا هذه القيمة؟ | أين تُستهلك؟ |
|---|---|---|---|
| IMAGE_SIZE | 224 | مقاس عين الطبيب المدرَّبة | input_shape وtarget_size |
| LEARNING_RATE | 0.001 | خطوة متوازنة (أكبر=تأرجح، أصغر=بطء) | Adam داخل compile |
| DROPOUT_RATE | 0.3 | ضمن المدى الشائع 0.2–0.5 | Dropout |
| BATCH_SIZE | 32 | توازن سرعة/ذاكرة | flow_from_directory |
| EPOCHS | 10 | سقف أعلى؛ الحارس قد يوقف قبله | fit |
| TRAIN/VAL/TEST_DIR | مسارات | مواقع تقسيم A2 | الأحزمة |
| MODEL_PATH | models/best_model.keras | مكان حفظ الأفضل | ModelCheckpoint |

### 2.3 الدالة build_model() — تبني وتجمّع
| السطر | النوع | الفائدة ولماذا |
|---|---|---|
| base_model = MobileNetV2(input_shape=(IMAGE_SIZE,IMAGE_SIZE,3), include_top=False, weights="imagenet") | استدعاء صنف → متغير | جلب الطبيب بلا رأسه (الـ1000 فئة لا تعنينا) وبخبرته |
| base_model.trainable = False | تعيين خاصية | التجميد: حماية خبرة الملايين؛ يظهر كـ Non-trainable |
| inputs = Input(shape=...) | استدعاء صنف → متغير | إعلان شكل الدخول — بداية الخط |
| x = base_model(inputs, training=False) | استدعاء كائن كدالة → متغير | عبور الطبيب؛ training=False يثبت طبقات تطبيعه |
| x = GlobalAveragePooling2D()(x) | صنف ثم تمرير (قوسان) | جدول 49 صفحة → ملخص 1280 |
| x = Dense(128, activation="relu")(x) | صنف ثم تمرير | المتعلم؛ 128 خلية حجم متوسط؛ relu تمرر المهم |
| x = Dropout(DROPOUT_RATE)(x) | صنف ثم تمرير | منع الحفظ؛ القيمة من الثابت |
| outputs = Dense(1, activation="sigmoid")(x) | صنف ثم تمرير → متغير | القاضي؛ خلية واحدة لثنائية التصنيف؛ sigmoid تحشر الناتج 0→1 |
| model = Model(inputs, outputs) | استدعاء صنف → متغير | يتتبع من الباب للقاضي ويغلفه |
| model.compile(optimizer=Adam(LEARNING_RATE), loss="binary_crossentropy", metrics=["accuracy"]) | دالة تابعة | قواعد اللعبة: كيف يصحح، كيف يقيس، ماذا نقرأ |
| return model | إرجاع | تسليم المنتج للمستدعي |

### 2.4 الدالة preprocess_image(image_path) — تجهيز صورة واحدة
| السطر | النوع | الفائدة ولماذا |
|---|---|---|
| img = tf.keras.utils.load_img(image_path, target_size=(IMAGE_SIZE,IMAGE_SIZE)) | دالة → متغير | تحميل+تحجيم بخطوة عبر PIL — استقلال عن OpenCV |
| arr = tf.keras.utils.img_to_array(img) | دالة → متغير | كائن صورة → مصفوفة أرقام (224,224,3) |
| arr = tf.expand_dims(arr, axis=0) | دالة | بُعد الدفعة → (1,224,224,3) |
| arr = preprocess_input(arr) | دالة مستوردة | القياس إلى [-1,1] لغة الطبيب |
| return arr | إرجاع | مصفوفة جاهزة للإطعام |

### 2.5 الدالة make_generators() — الأحزمة
| السطر | النوع | الفائدة ولماذا |
|---|---|---|
| train_gen = ImageDataGenerator(preprocessing_function=preprocess_input, rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1, horizontal_flip=True) | صنف → متغير | آلة تدريب: قياس + زيادة خفيفة لا تشوه السن |
| val_gen = ImageDataGenerator(preprocessing_function=preprocess_input) | صنف → متغير | آلة تحقق: قياس فقط — الامتحان لا يُحرَّف |
| train_loader = train_gen.flow_from_directory(TRAIN_DIR, target_size=..., batch_size=BATCH_SIZE, class_mode="binary") | دالة تابعة → متغير | الحزام: يمشي المجلد، يرقيم الفئات، يدفع 32 |
| val_loader = val_gen.flow_from_directory(VAL_DIR, ...) | دالة تابعة → متغير | حزام التحقق بلا زيادة |
| return train_loader, val_loader | إرجاع tuple | يُفك في train_model |

### 2.6 الدالة compute_class_weight(train_loader) — ميزان العدالة
| السطر | النوع | الفائدة ولماذا |
|---|---|---|
| labels = list(train_loader.classes) | خاصية + دالة مدمجة → متغير | classes تحمل رقم تصنيف كل صورة؛ نحوّلها قائمة للعد |
| n_caries = labels.count(0) / n_healthy = labels.count(1) / total | متغيرات عبر دالة count | أعداد الفئات: 219/121/340 |
| return {0: total/(2*n_caries), 1: total/(2*n_healthy)} | إرجاع قاموس | معادلة الموازنة: الأقلية وزنها أكبر فتصبح أخطاؤها أغلى |

### 2.7 الدالة make_callbacks() — كود التوقف
| السطر | النوع | الفائدة ولماذا |
|---|---|---|
| checkpoint = tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True) | صنف → متغير | المصور: يراقب val_loss ويكتب الملف فقط عند تحسن — حماية على القرص |
| early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True) | صنف → متغير | الحكم: 3 حقب بلا تحسين → يوقف؛ والاستعادة تعيد أوزان أفضل حقبة للذاكرة |
| return [checkpoint, early_stop] | إرجاع قائمة | تُسلَّم لـ fit |

كيف يعمل التوقف فعلياً؟
سطر التوصيل: callbacks=make_callbacks() داخل fit.
بعد كل حقبة تستدعي كيراس (من داخل المكتبة) on_epoch_end لكل حارس:
EarlyStopping يقارن val_loss بالأفضل ويعدّ؛ عند بلوغ 3 يضبط stop_training=True فتخرج الحلقة،
ثم تُستعاد أفضل أوزان؛ وCheckpoint كان حفظ اللقطة للقرص.

### 2.8 الدالة train_model() — قلب العملية
| السطر | النوع | الفائدة ولماذا |
|---|---|---|
| model = build_model() | استدعاء دالتنا → متغير | النموذج المجمع |
| train_loader, val_loader = make_generators() | فك tuple | الحزامان |
| class_weight = compute_class_weight(train_loader) | استدعاء → متغير قاموس | ميزان العدالة |
| print("Class weights:", ...) + سطر عربي | دالة مدمجة | تحقق مرئي ثنائي اللغة |
| history = model.fit(train_loader, epochs=EPOCHS, validation_data=val_loader, class_weight=class_weight, callbacks=make_callbacks()) | دالة تابعة → متغير | حلقة التدريب: 11 دفعة تعلم ← 3 دفعات val بلا تعلم ← الحارسان؛ history يسجل المنحنيات |
| return model, history | إرجاع tuple | للمستدعي |

### 2.9 الحارس if __name__ == "__main__": — يعمل عند التشغيل المباشر فقط
| السطر | النوع | الفائدة ولماذا |
|---|---|---|
| model, history = train_model() | استدعاء + فك | التدريب الكامل |
| test_gen = ImageDataGenerator(preprocessing_function=preprocess_input) | صنف → متغير | آلة اختبار بلا زيادة |
| test_loader = test_gen.flow_from_directory(TEST_DIR, ..., shuffle=False) | دالة تابعة → متغير | shuffle=False ترتيب ثابت: تقييم نزيه قابل لإعادة الحساب |
| test_loss, test_acc = model.evaluate(test_loader) | دالة تابعة → متغيران | الامتحان النهائي مرة واحدة |
| ستة أسطر print ثنائية اللغة | دوال مدمجة | النتيجة والمسار بشكل مقروء في طرفية ويندوز |

---

## 3) ملف src/verify_model.py — شهادة الاستقلال

فكرته: لا نصدق رقم التدريب لأنه طبعه؛ نعيد حسابه بطريقة مختلفة ومن ملف محفوظ.

| السطر | النوع | الفائدة ولماذا |
|---|---|---|
| الاستيرادات الثلاثة | أوامر استيراد | الملف مستقل بذاته: المحرك + دالة القياس + أمين المكتبة |
| MODEL_PATH = "models/best_model.keras" / TEST_DIR = "data/test" | ثابتان محليان | مسارا الشهادة |
| model = tf.keras.models.load_model(MODEL_PATH) | دالة → متغير | بناء النموذج من القرص لا من الذاكرة — يثبت صلاحية الملف المحفوظ |
| test_gen / test_loader | صنف + دالة تابعة → متغيران | حزام اختبار بلا زيادة وبترتيب ثابت |
| loss, acc = model.evaluate(test_loader) | دالة تابعة → متغيران | إعادة حساب الدقة بطريقة كيراس |
| probs = model.predict(test_loader, verbose=0) | دالة تابعة → متغير | احتمالات كل الصور (75 رقماً) |
| preds = (probs > 0.5).astype(int).flatten() | عمليات مصفوفات → متغير | القرار: فوق 0.5 = تسوس؛ تحويل لـ0/1؛ صف واحد |
| truth = test_loader.classes | خاصية → متغير | الحقيقة من أسماء المجلدات |
| correct = int((preds == truth).sum()) | مقارنة وعدّ → متغير | عدّ المطابقات صورة صورة — طريقة مستقلة عن دقة كيراس |
| سطرا print | دوال مدمجة | الشهادتان |

ناتجه الفعلي:
Reproduced test accuracy: 0.8666666746139526
Correct images: 65 / 75
مطابقة حرفية لرقم التدريب = النتيجة مثبتة.

---

## 4) ماذا حدث فعلياً + الإثباتات

منحنى التدريب:
| الحقبة | loss | accuracy | val_loss | val_accuracy |
|---|---|---|---|---|
| 1 | 0.6398 | 68.2% | 0.6572 | 67.6% |
| 2 | 0.5039 | 75.9% | 0.4650 | 81.7% |
| 3 | 0.3914 | 81.2% | 0.4133 (الأفضل) | 81.7% |
| 4 | 0.3323 | 86.5% | 0.4224 | 85.9% |
| 5 | 0.2843 | 87.9% | 0.4287 | 83.1% |
| 6 | 0.2683 | 89.1% | 0.4565 | 84.5% |

القصة: تعلم سريع حتى الحقبة 3 ← بعدها تدريب يتحسن وval يصعد = بدء حفظ ←
الحارس أوقف عند 6/10 (ثلاث حقب صبر) ← استُعيدت أوزان الحقبة 3 ← واللقطة على القرص.
الامتحان النهائي: خسارة 0.3348 ودقة 86.67% = 65/75.

طبقات الإثبات الثلاث:
1. حساب داخلي: 65÷75=0.8667؛ عدّ الدفعات 11 و3 يطابق 340 و75؛ معادلة الأوزان مطابقة؛ منطق patience مطابق
2. مراجعة كود: test لُمس مرة واحدة وبلا زيادة وبترتيب ثابت
3. إعادة إنتاج مستقلة: verify_model.py حمّل من القرص وعدّ صورة صورة فطابق

---

## 5) عقد الدمج للمرحلة 2 (لبلال)

- المدخل: صورة ملونة 224 → preprocess_input — بلا OpenCV
- الفئات: caries=0 / healthy=1؛ القرار: احتمال > 0.5
- النموذج: models/best_model.keras يُحمّل بـ load_model
- القادم: دالة predict تُرجع {class, confidence} ثم Grad-CAM

---

## 6) المسرد

Transfer Learning • Pretrained Base • Freezing • Feature Map • Global Average Pooling •
Dense • Dropout • Sigmoid • Batch • Epoch • Loss • Optimizer/Adam • Learning Rate •
Validation/Test • Augmentation • Class Weight • Overfitting • EarlyStopping •
ModelCheckpoint • restore_best_weights • class_indices • Docstring