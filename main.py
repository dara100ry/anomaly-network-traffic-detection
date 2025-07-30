from src.load_data import load_nsl_kdd, load_nsl_kdd_test
# from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

train_path = "./data/KDDTrain+.txt"
test_path = "./data/KDDTest+.txt"

X_train, y_train, encoders = load_nsl_kdd(train_path)
X_test, y_test = load_nsl_kdd_test(test_path, encoders)


# ۲. تقسیم داده‌ها به آموزش و تست (برای اعتبارسنجی اولیه)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ۳. تعریف و آموزش مدل Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train,y_train)

# ۴. پیش‌بینی و ارزیابی روی داده تست
y_pred = model.predict(X_test)
# 5. ارزیابی
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))




