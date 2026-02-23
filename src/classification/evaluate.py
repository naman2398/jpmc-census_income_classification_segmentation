import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve,
    precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay
)
from xgboost import XGBClassifier
import shap

OUT = "outputs/classification"
MODEL_DIR = "outputs/classification/models"


def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\n{'='*50}")
    print(f"{name}")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

    return y_pred, y_prob


def plot_roc_curves(results, y_test):
    plt.figure(figsize=(8, 6))
    for name, y_prob in results.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/roc_curve.png", dpi=150)
    plt.close()


def plot_precision_recall(results, y_test):
    plt.figure(figsize=(8, 6))
    for name, y_prob in results.items():
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        plt.plot(recall, precision, label=name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/precision_recall_curve.png", dpi=150)
    plt.close()


def plot_confusion(name, y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["<=50K", ">50K"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {name}")
    plt.tight_layout()
    plt.savefig(f"{OUT}/confusion_matrix_{name.lower().replace(' ', '_')}.png", dpi=150)
    plt.close()


def plot_feature_importance(model, preprocessor, numeric_cols, categorical_cols, top_n=20):
    cat_features = list(preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_cols))
    feature_names = numeric_cols + cat_features
    importance = model.feature_importances_
    top_idx = np.argsort(importance)[-top_n:]

    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), importance[top_idx])
    plt.yticks(range(top_n), [feature_names[i] for i in top_idx])
    plt.xlabel("Feature Importance")
    plt.title(f"Top {top_n} Features — XGBoost")
    plt.tight_layout()
    plt.savefig(f"{OUT}/feature_importance.png", dpi=150)
    plt.close()


def plot_shap_importance(model, X_test, preprocessor, numeric_cols, categorical_cols, top_n=20):
    cat_features = list(preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_cols))
    feature_names = numeric_cols + cat_features

    explainer = shap.TreeExplainer(model)
    X_sample = X_test[np.random.RandomState(42).choice(X_test.shape[0], min(5000, X_test.shape[0]), replace=False)]
    shap_values = explainer.shap_values(X_sample)

    for suffix, kwargs in [("summary", {"plot_type": "bar"}), ("beeswarm", {})]:
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                          max_display=top_n, show=False, plot_size=(12, 8), **kwargs)
        plt.title(f"Top {top_n} Features — SHAP {suffix.title()}", pad=20)
        plt.savefig(f"{OUT}/shap_{suffix}.png", dpi=150, bbox_inches="tight")
        plt.close()

    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[-top_n:][::-1]
    print(f"\nTop {top_n} features by mean |SHAP value|:")
    for i in top_idx:
        print(f"  {feature_names[i]}: {mean_abs[i]:.4f}")


def main():
    os.makedirs(OUT, exist_ok=True)

    # --- Load models and artifacts ---
    lr = joblib.load(f"{MODEL_DIR}/logistic_regression.joblib")
    xgb = XGBClassifier()
    xgb.load_model(f"{MODEL_DIR}/xgboost.json")
    preprocessor = joblib.load(f"{MODEL_DIR}/preprocessor.joblib")
    X_test = np.load(f"{MODEL_DIR}/X_test.npy")
    y_test = np.load(f"{MODEL_DIR}/y_test.npy")
    col_meta = joblib.load(f"{MODEL_DIR}/column_meta.joblib")
    numeric_cols = col_meta["numeric"]
    categorical_cols = col_meta["categorical"]

    print(f"Loaded models from {MODEL_DIR}/")
    print(f"Test set: {len(y_test)} samples")

    # --- Evaluate ---
    lr_pred, lr_prob = evaluate("Logistic Regression", lr, X_test, y_test)
    xgb_pred, xgb_prob = evaluate("XGBoost", xgb, X_test, y_test)

    # --- Plots ---
    results = {"Logistic Regression": lr_prob, "XGBoost": xgb_prob}
    plot_roc_curves(results, y_test)
    plot_precision_recall(results, y_test)
    plot_confusion("Logistic Regression", y_test, lr_pred)
    plot_confusion("XGBoost", y_test, xgb_pred)
    plot_feature_importance(xgb, preprocessor, numeric_cols, categorical_cols)
    plot_shap_importance(xgb, X_test, preprocessor, numeric_cols, categorical_cols)

    print(f"\nPlots saved to {OUT}/")


if __name__ == "__main__":
    main()
