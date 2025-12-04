# For Run Project
1.create venv (optional)

2.install requirements from "requirements.txt"

3.run "python -m src.train_model"

4.you can see Accuracy and calculations of model by run python "main.py"

5.run  "uvicorn backend.app:app --reload --port 8000" for backend

6.run  "streamlit run frontend/app.py" for frontend

7.now you can upload your csv file.

8.for project test you can use "test csv files" in data folder.
# 🛡️ Anomaly-based Intrusion Detection System using Machine Learning  
**سیستم تشخیص نفوذ در شبکه مبتنی بر یادگیری ماشین**
---

## 📘 مقدمه  
در این پروژه یک سیستم تشخیص نفوذ (IDS) مبتنی بر یادگیری ماشین توسعه داده شده است که قادر است ترافیک شبکه را تحلیل کرده و رفتارهای غیرعادی را شناسایی کند.

برای آموزش مدل از دیتاست استاندارد **NSL-KDD** استفاده شده و سه الگوریتم مورد مقایسه قرار گرفتند:

- 🌲 Random Forest  
- 🚀 XGBoost  
- 💡 LightGBM  

در نهایت الگوریتم **Random Forest** با دقت **89.34%** و مقدار **ROC-AUC = 0.968** به عنوان مدل نهایی انتخاب شد.

همچنین یک **FastAPI Backend** و یک **Streamlit Frontend** برای تحلیل آنلاین ترافیک و بارگذاری فایل‌های CSV طراحی شده است.

---

# 🏗️ معماری سیستم  

## 🔧 اجزای اصلی

### 1. **Model Trainer**
- آموزش مدل‌ها  
- پیش‌پردازش داده‌ها  
- ذخیره مدل در قالب `model.pkl`  

### 2. **Backend – FastAPI**
- دریافت رکورد یا فایل CSV  
- اجرای مدل  
- برگرداندن برچسب نهایی و میزان اطمینان مدل  

### 3. **Frontend – Streamlit**
- رابط کاربری  
- آپلود فایل CSV  
- نمایش خروجی‌ها به صورت رنگی  

---

# 📊 دیاگرام‌ها (UML)

## 📌 Use Case Diagram
- ورود فایل ترافیک توسط کاربر  
- پردازش داده  
- نمایش نتیجه  

## 📌 Class Diagram (مفاهیم اصلی)

| کلاس | توضیح |
|------|--------|
| `DataLoader` | بارگذاری و پاکسازی داده‌ها |
| `Preprocessor` | نرمال‌سازی – One-Hot Encoding |
| `ModelTrainer` | آموزش مدل RandomForest |
| `ModelPredictor` | پیش‌بینی با مدل ذخیره‌شده |
| `FastAPIService` | سرویس‌دهی از طریق API |
| `StreamlitUI` | رابط گرافیکی |
| `User` | کاربر سیستم |

## 📌 Sequence Diagram  
نمایش توالی کامل از آپلود داده تا پیش‌بینی مدل.

---

# ⚙️ پیش‌پردازش داده‌ها

### 🔸 ویژگی‌های عددی
مثل:  
`src_bytes`, `dst_bytes`

### 🔸 ویژگی‌های متنی
مثل:  
`protocol_type`, `service`, `flag`

### 🔸 استفاده از ColumnTransformer  
- One-Hot Encoding برای ویژگی‌های متنی  
- StandardScaler برای ویژگی‌های عددی  

### 🗑️ حذف ویژگی‌های کم‌اهمیت  
ویژگی‌های زیر حذف شدند:

dst_host_diff_srv_rate
hot
dst_host_srv_count
count
num_outbound_cmds # همیشه صفر
is_host_login # تقریباً همیشه صفر
land # بسیار نادر

yaml
Copy code

این حذف باعث **بهبود دقت نهایی** شده است.

برای پیدا کردن ویژگی‌های کم‌اهمیت از فایل `feature_ablation.py` استفاده شده است.

---

# 🤖 آموزش مدل (Model Training)

هر سه مدل زیر آموزش داده شدند:

- RandomForestClassifier  
- XGBoostClassifier  
- LightGBMClassifier  

یک **Pipeline** یکپارچه شامل پیش‌پردازش + مدل ساخته شد.

### 🔥 انتخاب مدل نهایی  
نتایج نشان داد:

> 📌 **RandomForest بهترین عملکرد را داشت**

### ⚙️ تنظیمات نهایی RandomForest

```python
RandomForestClassifier(
    n_estimators=2000,
    max_depth=None,
    class_weight="balanced_subsample",
    max_features='sqrt',
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)
🎯 یافتن بهترین Threshold
تابع find_optimal_threshold روی احتمالات خروجی مدل اجرا شده و بهترین آستانه برای بیشترین دقت انتخاب شده است.

💾 ذخیره مدل
مدل نهایی همراه با:

pipeline

threshold

accuracy

کلاس‌ها

attack_idx

در فایل:

bash
Copy code
models/model.pkl
ذخیره می‌شود.

🖥️ Backend – FastAPI
📌 پیش‌بینی یک رکورد JSON
python
Copy code
@app.post("/predict-json")
def predict_json(record: dict):
    df = pd.DataFrame([record])
    proba = pipeline.predict_proba(df)[0][attack_idx]
    label = "attack" if proba >= threshold else "normal"
    return {"label": label, "confidence": float(proba)}
📌 پیش‌بینی فایل CSV
هر رکورد پردازش شده و خروجی به صورت لیست برگردانده می‌شود.

🖥️ Frontend – Streamlit
امکانات:
آپلود CSV

پردازش رکوردها از طریق API

نمایش نتیجه با رنگ‌بندی

نمایش میزان اطمینان مدل

ارتباط با API:

python
Copy code
api_url = st.text_input("API URL", value="http://127.0.0.1:8000")
🧪 معیارهای ارزیابی مدل
Accuracy

Precision

Recall (TPR)

F1-score

Specificity

False Positive Rate

ROC-AUC

✅ نتیجه‌گیری
در این پروژه:

مدل RandomForest بهترین عملکرد را ارائه داد.

معماری ماژولار سیستم باعث امکان استقرار سریع در محیط عملیاتی شد.

با حذف ویژگی‌های کم‌اهمیت، دقت بهتر شد.

امکان تحلیل دادهٔ واقعی از طریق FastAPI + Streamlit UI فراهم شد.

این سیستم قابلیت توسعه برای تشخیص نفوذ در زمان واقعی (Real-Time IDS) را دارد و در نسخه‌های بعدی می‌تواند سبک‌تر و سریع‌تر نیز شود.

