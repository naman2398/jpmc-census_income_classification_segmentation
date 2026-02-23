import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from data_loader import load_data, prepare_features

MODEL_DIR = "outputs/classification/models"


def build_preprocessor(numeric_cols, categorical_cols):
    return ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", sparse_output=False, min_frequency=50), categorical_cols)
    ])


def temporal_split(feature_df, label, weight, year, preprocessor):
    mask_94 = year == 94
    mask_95 = year == 95

    idx_95 = feature_df.index[mask_95]
    idx_95_train, idx_95_test = train_test_split(
        idx_95, test_size=0.4, random_state=42, stratify=label[mask_95]
    )

    train_idx = feature_df.index[mask_94].append(idx_95_train)
    test_idx = idx_95_test

    X_train = preprocessor.fit_transform(feature_df.loc[train_idx])
    X_test = preprocessor.transform(feature_df.loc[test_idx])
    y_train = label.loc[train_idx].values
    y_test = label.loc[test_idx].values
    w_train = weight.loc[train_idx].values

    return X_train, X_test, y_train, y_test, w_train


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = load_data()
    feature_df, numeric_cols, categorical_cols, label, weight, year = prepare_features(df)

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    X_train, X_test, y_train, y_test, w_train = temporal_split(
        feature_df, label, weight, year, preprocessor
    )

    pos_ratio = y_train.sum() / len(y_train)
    scale_pos_weight = (1 - pos_ratio) / pos_ratio
    print(f"Train: {len(y_train)} samples | Test: {len(y_test)} samples | Positive ratio: {pos_ratio:.3f}")

    # --- Logistic Regression ---
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    lr.fit(X_train, y_train, sample_weight=w_train)

    # --- XGBoost ---
    xgb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        scale_pos_weight=scale_pos_weight, eval_metric="auc",
        base_score=0.5, random_state=42, n_jobs=-1
    )
    xgb.fit(X_train, y_train, sample_weight=w_train)

    # --- Save models and artifacts ---
    joblib.dump(lr, f"{MODEL_DIR}/logistic_regression.joblib")
    xgb.save_model(f"{MODEL_DIR}/xgboost.json")
    joblib.dump(preprocessor, f"{MODEL_DIR}/preprocessor.joblib")
    np.save(f"{MODEL_DIR}/X_test.npy", X_test)
    np.save(f"{MODEL_DIR}/y_test.npy", y_test)
    joblib.dump({"numeric": numeric_cols, "categorical": categorical_cols}, f"{MODEL_DIR}/column_meta.joblib")

    print(f"Models and artifacts saved to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
