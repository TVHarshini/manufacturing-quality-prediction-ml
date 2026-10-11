"""
Manufacturing Quality Prediction using Machine Learning
Educational portfolio project using SIMULATED data.

Important: This project demonstrates a reproducible ML workflow. The data are
synthetically generated for practice and are NOT real factory measurements.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay,
    RocCurveDisplay
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
RANDOM_STATE = 42

def create_simulated_dataset(n=1200, random_state=RANDOM_STATE):
    """Create illustrative process measurements and a simulated defect label."""
    rng = np.random.default_rng(random_state)
    temperature = rng.normal(185, 18, n).clip(130, 240)
    pressure = rng.normal(7.5, 1.25, n).clip(3.5, 11.5)
    machine_speed = rng.normal(120, 22, n).clip(55, 185)
    vibration = rng.gamma(shape=2.0, scale=0.85, size=n).clip(0.05, 8.0)
    material_moisture = rng.normal(4.0, 1.1, n).clip(0.3, 8.5)

    # Illustrative rule with randomness; not a validated engineering model.
    risk = (
        0.045 * np.maximum(temperature - 195, 0)
        + 0.48 * np.maximum(6.5 - pressure, 0)
        + 0.026 * np.maximum(machine_speed - 135, 0)
        + 0.62 * np.maximum(vibration - 1.7, 0)
        + 0.20 * np.maximum(material_moisture - 4.5, 0)
        + rng.normal(0, 0.65, n)
    )
    probability = 1 / (1 + np.exp(-(risk - 1.7)))
    defect = rng.binomial(1, probability)

    df = pd.DataFrame({
        "temperature_c": temperature.round(2),
        "pressure_bar": pressure.round(2),
        "machine_speed_rpm": machine_speed.round(2),
        "vibration_mm_s": vibration.round(3),
        "material_moisture_pct": material_moisture.round(2),
        "defect": defect
    })
    return df

def main():
    df = create_simulated_dataset()
    data_path = OUT / "simulated_manufacturing_data.csv"
    df.to_csv(data_path, index=False)

    X = df.drop(columns="defect")
    y = df["defect"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=250, max_depth=8, min_samples_leaf=3,
            class_weight="balanced", random_state=RANDOM_STATE
        ))
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    metrics = {
        "dataset_type": "SIMULATED / educational only",
        "rows_total": int(len(df)),
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "defect_rate_percent": round(float(y.mean() * 100), 2),
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "precision_defect_class": round(float(precision_score(y_test, pred, zero_division=0)), 4),
        "recall_defect_class": round(float(recall_score(y_test, pred, zero_division=0)), 4),
        "f1_defect_class": round(float(f1_score(y_test, pred, zero_division=0)), 4),
        "random_state": RANDOM_STATE,
        "note": "Metrics describe this simulated dataset only; they do not establish real-world factory performance."
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (OUT / "classification_report.txt").write_text(
        "CLASSIFICATION REPORT — SIMULATED DATA ONLY\n\n" +
        classification_report(y_test, pred, target_names=["No defect", "Defect"], zero_division=0) +
        "\nCaution: results are from simulated educational data, not real factory data.\n",
        encoding="utf-8"
    )

    # Figure 1: class distribution
    fig, ax = plt.subplots(figsize=(7, 4.5))
    counts = df["defect"].map({0: "No defect", 1: "Defect"}).value_counts().reindex(["No defect", "Defect"])
    counts.plot(kind="bar", ax=ax)
    ax.set_title("Simulated Dataset: Quality Label Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of samples")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(OUT / "class_distribution.png", dpi=160)
    plt.close(fig)

    # Figure 2: confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, pred, display_labels=["No defect", "Defect"],
        cmap="Blues", values_format="d", ax=ax
    )
    ax.set_title("Confusion Matrix — Test Set (Simulated Data)")
    fig.tight_layout()
    fig.savefig(OUT / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    # Figure 3: feature importance
    clf = model.named_steps["classifier"]
    importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    importances.plot(kind="barh", ax=ax)
    ax.set_title("Random Forest Feature Importance")
    ax.set_xlabel("Relative importance")
    ax.set_ylabel("Input feature")
    fig.tight_layout()
    fig.savefig(OUT / "feature_importance.png", dpi=160)
    plt.close(fig)

    # Figure 4: ROC curve
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, probs, ax=ax)
    ax.set_title("ROC Curve — Test Set (Simulated Data)")
    fig.tight_layout()
    fig.savefig(OUT / "roc_curve.png", dpi=160)
    plt.close(fig)

    # Screenshot-like portfolio summary image, created from actual run metrics.
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")
    lines = [
        "MANUFACTURING QUALITY PREDICTION",
        "Python + scikit-learn | Reproducible educational demo",
        "",
        f"Dataset rows: {metrics['rows_total']}   |   Train: {metrics['training_rows']}   |   Test: {metrics['test_rows']}",
        f"Simulated defect rate: {metrics['defect_rate_percent']}%",
        "",
        f"Accuracy:  {metrics['accuracy']:.3f}",
        f"Precision: {metrics['precision_defect_class']:.3f}",
        f"Recall:    {metrics['recall_defect_class']:.3f}",
        f"F1-score:  {metrics['f1_defect_class']:.3f}",
        "",
        "Workflow: generate data → split → train Random Forest → evaluate",
        "",
        "IMPORTANT: Data are simulated for learning. Not validated on factory data."
    ]
    ax.text(0.05, 0.95, "\n".join(lines), va="top", ha="left",
            family="monospace", fontsize=13,
            bbox=dict(boxstyle="round,pad=1", facecolor="white", edgecolor="black"))
    fig.tight_layout()
    fig.savefig(ROOT / "screenshots" / "run_results_summary.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("Manufacturing Quality Prediction — completed")
    print("Dataset: SIMULATED educational data")
    print(f"Rows: {len(df)} | Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision_defect_class']:.4f}")
    print(f"Recall:    {metrics['recall_defect_class']:.4f}")
    print(f"F1-score:  {metrics['f1_defect_class']:.4f}")
    print("Files saved in outputs/ and screenshots/")

if __name__ == "__main__":
    main()
