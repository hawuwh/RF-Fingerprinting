import gc
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from tensorflow.keras.models import load_model

# --- Configuration ---
DATA_PATH = "processed_wisig.npz"
MODEL_PATH = "best_rf_model.keras"
REPORT_FILENAME = "final_performance_report.txt"
CM_FILENAME = "confusion_matrix.png"


def generate_evaluation():
    print("--- Starting System Evaluation ---")

    # Load Test Data Only
    data = np.load(DATA_PATH)
    X_test = data["X_test"]
    y_test = data["y_test"]
    del data
    gc.collect()

    # Load Trained Model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Trained model not found.")

    model = load_model(MODEL_PATH)

    # Perform Inference
    print(" [Status] Running inference on test set...")
    y_pred_probs = model.predict(X_test, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # --- 1. Calculate Standard Metrics ---
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted"
    )

    print("\n" + "=" * 40)
    print(f"PERFORMANCE SUMMARY")
    print("=" * 40)
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("=" * 40)

    # --- 2. Generate Detailed Report ---
    print(f" [Status] Saving report to {REPORT_FILENAME}...")
    report = classification_report(y_true, y_pred, digits=4)
    with open(REPORT_FILENAME, "w") as f:
        f.write("RF Fingerprinting System - Performance Report\n")
        f.write("=============================================\n\n")
        f.write(f"Overall Accuracy: {acc * 100:.2f}%\n")
        f.write(f"Weighted F1-Score: {f1:.4f}\n\n")
        f.write(report)

    # --- 3. Generate Confusion Matrix ---
    print(f" [Status] Generating Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(12, 10))
    # Plotting subset of first 20 devices for readability
    sns.heatmap(cm[:20, :20], annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix (Subset of First 20 Devices)")
    plt.ylabel("Actual Device ID")
    plt.xlabel("Predicted Device ID")
    plt.savefig(CM_FILENAME)
    print(f" [Success] Evaluation complete. Artifacts saved.")


if __name__ == "__main__":
    generate_evaluation()
