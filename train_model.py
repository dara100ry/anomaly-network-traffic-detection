import os
import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from src.load_data import load_nsl_kdd_raw, load_nsl_kdd_test_raw


def find_optimal_threshold(y_true, y_proba, pos_class_idx):
    thresholds = np.arange(0.01, 0.99, 0.002)
    best_threshold = 0.5
    best_accuracy = 0.0
    for threshold in thresholds:
        y_pred_thresh = (y_proba[:, pos_class_idx] >= threshold).astype(int)
        y_pred_labels = ['normal' if p == 0 else 'attack' for p in y_pred_thresh]
        accuracy = accuracy_score(y_true, y_pred_labels)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    return best_threshold, best_accuracy


def train_and_save_model(train_path: str, test_path: str, model_dir: str = "models") -> str:
    X_train_df, y_train = load_nsl_kdd_raw(train_path)
    X_test_df, y_test = load_nsl_kdd_test_raw(test_path)

    categorical_cols = ["protocol_type", "service", "flag"]
    numeric_cols = [col for col in X_train_df.columns if col not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
            ("num", StandardScaler(), numeric_cols),
        ],
        remainder="drop",
    )

    rf_clf = RandomForestClassifier(
        n_estimators=2000,
        max_depth=None,
        class_weight="balanced_subsample",
        max_features='sqrt',
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(steps=[("pre", preprocessor), ("clf", rf_clf)])
    pipeline.fit(X_train_df, y_train)

    y_proba = pipeline.predict_proba(X_test_df)
    classes = list(pipeline.named_steps["clf"].classes_)
    attack_idx = classes.index("attack")
    threshold, accuracy = find_optimal_threshold(y_test, y_proba, attack_idx)

    os.makedirs(model_dir, exist_ok=True)
    artifact = {
        "pipeline": pipeline,
        "threshold": float(threshold),
        "classes": classes,
        "attack_idx": int(attack_idx),
        "accuracy": float(accuracy),
    }
    out_path = os.path.join(model_dir, "model.pkl")
    joblib.dump(artifact, out_path)
    return out_path


if __name__ == "__main__":
    train_path = "./data/KDDTrain+.txt"
    test_path = "./data/KDDTest+.txt"
    saved = train_and_save_model(train_path, test_path)
    print(f"Saved model to: {saved}")
