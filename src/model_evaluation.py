"""
Model Evaluation, Visualization, and Test Set Validation Engine
Course: IT3051 - Fundamentals of Data Mining (SLIIT)
Project: Predictive Maintenance Diagnostic System (AI4I-PMDI Dataset)
Group: 05 - 'Cognita'

This module generates production-grade visualizations, performs final evaluation
on the untouched test set, logs experiment results, and exports production pipelines.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    balanced_accuracy_score,
    accuracy_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from src.model_training import TARGET_CLASSES

# Set styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10
FIGURES_DIR = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def plot_model_comparison(leaderboard_df: pd.DataFrame, save_path: Optional[Path] = None) -> Path:
    """Plots comparative bar chart of Macro F1, Macro Recall, and Balanced Accuracy across models."""
    if save_path is None:
        save_path = FIGURES_DIR / "01_model_comparison_metrics.png"

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    models = leaderboard_df["Model"].tolist()
    x = np.arange(len(models))
    width = 0.25

    macro_f1 = leaderboard_df["Macro F1 (Raw)"].tolist()
    macro_rec = leaderboard_df["Macro Recall (Raw)"].tolist()
    bal_acc = leaderboard_df["Balanced Accuracy (Raw)"].tolist()

    rects1 = ax.bar(x - width, macro_f1, width, label="Macro F1 (Primary)", color="#1f77b4")
    rects2 = ax.bar(x, macro_rec, width, label="Macro Recall", color="#ff7f0e")
    rects3 = ax.bar(x + width, bal_acc, width, label="Balanced Accuracy", color="#2ca02c")

    ax.set_ylabel("Score (0.0 to 1.0)", fontsize=11, fontweight="bold")
    ax.set_title("Stage 6: Cross-Validation Performance Across 5 Diverse Algorithms\n(AI4I-PMDI Multiclass Predictive Maintenance)", fontsize=12, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right", fontsize=10)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower right", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Add value labels
    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.2f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path


def plot_per_class_f1(baseline_results: Dict[str, Any], save_path: Optional[Path] = None) -> Path:
    """Plots grouped bar chart of Per-Class F1 across models, emphasizing rare failure detection."""
    if save_path is None:
        save_path = FIGURES_DIR / "02_per_class_f1_comparison.png"

    records = []
    for model_name, res in baseline_results.items():
        rep = res["classification_report"]
        for cls_name in TARGET_CLASSES:
            records.append({
                "Model": model_name,
                "Class": cls_name.replace(" Failure", "").replace(" Failures", ""),
                "F1 Score": rep[cls_name]["f1-score"],
            })
    df_cls = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    palette = sns.color_palette("muted", n_colors=len(baseline_results))
    sns.barplot(data=df_cls, x="Class", y="F1 Score", hue="Model", palette=palette, ax=ax)

    ax.set_title("Per-Class F1-Score Across Models (Minority Failure Diagnosis Sensitivity)", fontsize=12, fontweight="bold", pad=15)
    ax.set_xlabel("Diagnostic Class", fontsize=11, fontweight="bold")
    ax.set_ylabel("F1 Score", fontsize=11, fontweight="bold")
    ax.set_ylim(0.0, 1.05)
    ax.legend(title="Algorithm", loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path


def plot_feature_ablation(ablation_df: pd.DataFrame, save_path: Optional[Path] = None) -> Path:
    """Visualizes feature ablation: Raw features vs Domain Physics features."""
    if save_path is None:
        save_path = FIGURES_DIR / "03_feature_ablation_comparison.png"

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    
    models = ablation_df["Model"].tolist()
    x = np.arange(len(models))
    width = 0.35

    exp_a_raw = [float(val.split(" ")[0]) for val in ablation_df["Exp A (Raw) Macro F1"]]
    exp_b_phys = [float(val.split(" ")[0]) for val in ablation_df["Exp B (Physics) Macro F1"]]

    r1 = ax.bar(x - width/2, exp_a_raw, width, label="Exp A: Raw Features Only (16 cols)", color="#7f7f7f")
    r2 = ax.bar(x + width/2, exp_b_phys, width, label="Exp B: Raw + Domain Physics Features (23 cols)", color="#2ca02c")

    ax.set_ylabel("Macro F1-Score", fontsize=11, fontweight="bold")
    ax.set_title("Stage 7: Impact of Physics-Informed Feature Engineering on Macro-F1\n(Capturing Spindle Power, Delta T, and Overstrain Mechanics)", fontsize=11, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0.0, 0.85)
    ax.legend(loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    for rects in [r1, r2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.4f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path


def plot_imbalance_comparison(imbalance_df: pd.DataFrame, save_path: Optional[Path] = None) -> Path:
    """Visualizes class imbalance strategies comparison."""
    if save_path is None:
        save_path = FIGURES_DIR / "04_imbalance_strategy_comparison.png"

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    strategies = imbalance_df["Imbalance Strategy"].tolist()
    macro_f1 = imbalance_df["Macro F1 (Raw)"].tolist()
    macro_rec = imbalance_df["Macro Recall (Raw)"].tolist()

    x = np.arange(len(strategies))
    width = 0.35

    r1 = ax.bar(x - width/2, macro_f1, width, label="Macro F1", color="#1f77b4")
    r2 = ax.bar(x + width/2, macro_rec, width, label="Macro Recall", color="#ff7f0e")

    ax.set_ylabel("Score", fontsize=11, fontweight="bold")
    ax.set_title("Class Imbalance Strategy Comparison (Random Forest)\n(Algorithmic Cost-Weighting vs SMOTE vs Unweighted)", fontsize=11, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, fontsize=9.5)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    for rects in [r1, r2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.3f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path


def plot_baseline_vs_tuned(tuning_comparison_df: pd.DataFrame, save_path: Optional[Path] = None) -> Path:
    """Plots baseline vs tuned model performance comparison."""
    if save_path is None:
        save_path = FIGURES_DIR / "05_baseline_vs_tuned_comparison.png"

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    models = tuning_comparison_df["Model"].tolist()
    x = np.arange(len(models))
    width = 0.35

    base_f1 = tuning_comparison_df["Baseline Macro F1"].tolist()
    tuned_f1 = tuning_comparison_df["Tuned Macro F1"].tolist()

    r1 = ax.bar(x - width/2, base_f1, width, label="Baseline Model", color="#9ecae1")
    r2 = ax.bar(x + width/2, tuned_f1, width, label="Tuned Model (Hyperparameter Optimized)", color="#08519c")

    ax.set_ylabel("Macro F1-Score", fontsize=11, fontweight="bold")
    ax.set_title("Stage 7: Hyperparameter Tuning Performance Uplift (Macro-F1)", fontsize=11, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10.5)
    ax.set_ylim(0.0, 0.85)
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    for rects in [r1, r2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.4f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path


def plot_confusion_matrices(y_true: pd.Series, y_pred: np.ndarray, labels: List[str], save_path: Optional[Path] = None) -> Path:
    """Plots side-by-side Raw Count and Normalized Recall Confusion Matrix heatmaps."""
    if save_path is None:
        save_path = FIGURES_DIR / "06_final_confusion_matrices.png"

    cm_raw = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")

    short_labels = [l.replace(" Failure", "").replace(" Failures", "") for l in labels]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), dpi=300)

    # Raw counts
    sns.heatmap(cm_raw, annot=True, fmt="d", cmap="Blues", xticklabels=short_labels, yticklabels=short_labels, ax=axes[0])
    axes[0].set_title("Untouched Test Set: Raw Instance Confusion Matrix", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Predicted Condition", fontsize=10, fontweight="bold")
    axes[0].set_ylabel("True Condition", fontsize=10, fontweight="bold")

    # Normalized recall
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Greens", xticklabels=short_labels, yticklabels=short_labels, ax=axes[1])
    axes[1].set_title("Untouched Test Set: Normalized Recall Matrix (Sensitivity %)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Predicted Condition", fontsize=10, fontweight="bold")
    axes[1].set_ylabel("True Condition", fontsize=10, fontweight="bold")

    plt.suptitle("Final Champion Model Evaluation on Held-Out Test Set (N=2,000 Unseen Telemetry Snapshots)", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 12, save_path: Optional[Path] = None) -> Path:
    """Plots horizontal bar chart of top Permutation Feature Importances."""
    if save_path is None:
        save_path = FIGURES_DIR / "07_feature_importance_champion.png"

    top_df = importance_df.head(top_n).sort_values(by="Importance Mean", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    colors = ["#2ca02c" if "num__" in f and any(p in f for p in ["Temp_Difference", "Power", "Overstrain", "Missing"]) else "#1f77b4" for f in top_df["Feature"]]
    
    clean_names = [
        f.replace("num__", "").replace("cat__", "Category: ") for f in top_df["Feature"]
    ]

    ax.barh(clean_names, top_df["Importance Mean"], xerr=top_df["Importance Std"], color=colors, capsize=4)
    ax.set_xlabel("Mean Permutation Importance (Macro-F1 Loss on Shuffle)", fontsize=11, fontweight="bold")
    ax.set_title(f"Top {top_n} Predictive Features in Champion Model\n(Green = Domain Physics Engineered Features | Blue = Raw Telemetry & Operating Modes)", fontsize=11, fontweight="bold", pad=15)
    ax.grid(axis="x", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path


def evaluate_final_test_set(
    champion_model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    target_names: Optional[List[str]] = None,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Evaluates the chosen champion model strictly ONCE on the untouched test partition.
    
    Args:
        champion_model: Fitted final model.
        X_test: Test features (2,000 instances).
        y_test: Test target labels.
        target_names: List of class labels in order.
        
    Returns:
        Tuple of (test_metrics_dict, per_class_metrics_df).
    """
    if target_names is None:
        target_names = TARGET_CLASSES

    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.iloc[:, 0]

    y_pred = champion_model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred, labels=target_names)
    rep_dict = classification_report(
        y_test, y_pred, labels=target_names, target_names=target_names, output_dict=True, zero_division=0
    )

    metrics = {
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "Balanced Accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "Macro F1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "Macro Precision": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "Macro Recall": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "Weighted F1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "Weighted Precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "Weighted Recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "y_pred": y_pred,
        "confusion_matrix": cm,
        "classification_report": rep_dict,
    }

    per_class_records = []
    for cls in target_names:
        per_class_records.append({
            "Diagnostic Class": cls,
            "Precision": rep_dict[cls]["precision"],
            "Recall": rep_dict[cls]["recall"],
            "F1-Score": rep_dict[cls]["f1-score"],
            "Support (Test Count)": rep_dict[cls]["support"],
        })
    per_class_df = pd.DataFrame(per_class_records)

    return metrics, per_class_df
