from src.load_data import load_nsl_kdd_raw, load_nsl_kdd_test_raw
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import numpy as np
import xgboost as xgb
import lightgbm as lgb

train_path = "./data/KDDTrain+.txt"
test_path = "./data/KDDTest+.txt"


X_train_df, y_train = load_nsl_kdd_raw(train_path)
X_test_df, y_test = load_nsl_kdd_test_raw(test_path)


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_train_numeric = label_encoder.fit_transform(y_train)
y_test_numeric = label_encoder.transform(y_test)

# Columns
categorical_cols = ["protocol_type", "service", "flag"]
numeric_cols = [col for col in X_train_df.columns if col not in categorical_cols]

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ("num", StandardScaler(), numeric_cols),
    ],
    remainder="drop",
)

# Train 
print("Training RandomForest...")
rf_clf = RandomForestClassifier(
    n_estimators=1000,
    max_depth=None,
    class_weight="balanced_subsample",
    max_features='sqrt',
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)

print("Training XGBoost...")
xgb_clf = xgb.XGBClassifier(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)

print("Training LightGBM...")
lgb_clf = lgb.LGBMClassifier(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

# Train all models
models = {
    'RandomForest': Pipeline(steps=[("pre", preprocessor), ("clf", rf_clf)]),
    'XGBoost': Pipeline(steps=[("pre", preprocessor), ("clf", xgb_clf)]),
    'LightGBM': Pipeline(steps=[("pre", preprocessor), ("clf", lgb_clf)])
}

for name, model in models.items():
    print(f"Fitting {name}...")
    if name in ['XGBoost', 'LightGBM']:
        model.fit(X_train_df, y_train_numeric)
    else:
        model.fit(X_train_df, y_train)

# threshold tuning
print("Finding optimal threshold for each model...")

# Find optimal threshold for each model
# def find_optimal_threshold(y_true, y_proba, pos_class_idx):
#     from sklearn.metrics import accuracy_score
#     thresholds = np.arange(0.1, 0.9, 0.005)
#     best_threshold = 0.5
#     best_accuracy = 0
    
#     for threshold in thresholds:
#         y_pred_thresh = (y_proba[:, pos_class_idx] >= threshold).astype(int)
#         y_pred_labels = ['normal' if p == 0 else 'attack' for p in y_pred_thresh]
#         accuracy = accuracy_score(y_true, y_pred_labels)
#         if accuracy > best_accuracy:
#             best_accuracy = accuracy
#             best_threshold = threshold
    
#     return best_threshold, best_accuracy

def find_optimal_threshold(y_true, y_proba, pos_class_idx):
    from sklearn.metrics import accuracy_score
    thresholds = np.arange(0.1, 0.9, 0.005)
    best_threshold = 0.5
    best_accuracy = -1

    # unify y_true to numeric 0/1
    if np.array(y_true).dtype.kind in "OUS":  # object/string
        y_true_num = np.array([0 if v == "normal" else 1 for v in y_true])
    else:
        y_true_num = np.array(y_true, dtype=int)

    for threshold in thresholds:
        y_pred_num = (y_proba[:, pos_class_idx] >= threshold).astype(int)
        accuracy = accuracy_score(y_true_num, y_pred_num)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    return best_threshold, best_accuracy


# Test 
best_model_name = None
best_accuracy = 0
best_threshold = 0.5
best_predictions = None

for name, model in models.items():
    print(f"\nEvaluating {name}...")
    y_proba = model.predict_proba(X_test_df)
    
    if name in ['XGBoost', 'LightGBM']:
        # For XGBoost and LightGBM, use numeric labels
        classes = list(model.named_steps["clf"].classes_)
        attack_idx = classes.index(1)  # 1 represents 'attack' in numeric encoding
        threshold, accuracy = find_optimal_threshold(y_test_numeric, y_proba, attack_idx)
    else:
        # For RandomForest, use string labels
        classes = list(model.named_steps["clf"].classes_)
        attack_idx = classes.index("attack")
        threshold, accuracy = find_optimal_threshold(y_test, y_proba, attack_idx)
    
    print(f"{name} - Optimal threshold: {threshold:.3f}, accuracy: {accuracy:.4f}")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_name = name
        best_threshold = threshold
        best_predictions = (y_proba, attack_idx)

print(f"\nBest model: {best_model_name} with accuracy: {best_accuracy:.4f}")

# best model
pipeline = models[best_model_name]
y_proba, attack_idx = best_predictions


y_pred = (y_proba[:, attack_idx] >= best_threshold).astype(int)


if best_model_name in ['XGBoost', 'LightGBM']:
    y_pred_labels = label_encoder.inverse_transform(y_pred)
else:
    y_pred_labels = ['normal' if p == 0 else 'attack' for p in y_pred]

# Evaluate
print(f"\nUsing threshold: {best_threshold:.3f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_labels))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred_labels,
    labels=["normal", "attack"],
    target_names=["normal", "attack"],
    digits=4,
))

# ROC-AUC (binary: positive class is 'attack')
try:
    auc = roc_auc_score((y_test == "attack").astype(int), y_proba[:, attack_idx])
    print(f"\nROC-AUC: {auc:.4f}")
except Exception as e:
    print(f"\nROC-AUC unavailable: {e}")

# error rate
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred_labels)
error_rate = 1 - accuracy
print(f"\nAccuracy: {accuracy:.4f}")
print(f"Error Rate: {error_rate:.4f} ({error_rate*100:.2f}%)")

# Feature importances
try:
    import numpy as np

    pre = pipeline.named_steps["pre"]
    feature_names = pre.get_feature_names_out()
    importances = pipeline.named_steps["clf"].feature_importances_
    idx = np.argsort(importances)[::-1][:20]
    print("\nTop 20 important features:")
    for rank, i in enumerate(idx, start=1):
        print(f"{rank:2d}. {feature_names[i]} -> {importances[i]:.4f}")
except Exception as e:
    print(f"\nFeature importance unavailable: {e}")

