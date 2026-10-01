"""
Model Training and Cross-Validation Evaluation Engine
Course: IT3051 - Fundamentals of Data Mining (SLIIT)
Project: Predictive Maintenance Diagnostic System (AI4I-PMDI Dataset)
Group: 05 - 'Cognita'

This module implements production-grade model training, cross-validation evaluation,
and systematic metric calculation across diverse machine learning algorithms:
1. Multinomial Logistic Regression (Regularized Linear Baseline)
2. Random Forest Classifier (Nonlinear Bagging Ensemble)
3. HistGradientBoostingClassifier (Sequential Gradient Boosting)
4. Extra Trees Classifier (Extremely Randomized Bagging)
5. Support Vector Classifier (Cost-Sensitive RBF Kernel)
"""

import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.svm import SVC
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    accuracy_score,
)

TARGET_CLASSES = [
    "No failure",
    "Heat Dissipation Failure",
    "Overstrain Failure",
    "Power Failure",
    "Tool Wear Failure",
    "Random Failures",
]


def get_baseline_models(random_state: int = 42) -> Dict[str, Any]:
    """Instantiates five mathematically diverse classifiers with balanced class weighting.
    
    Algorithms:
    - Logistic Regression: Parametric linear baseline with L2 penalty (requires scaling).
    - Random Forest: Non-linear bagging ensemble of decorrelated deep trees (scale invariant).
    - HistGradientBoosting: Sequential histogram-based boosting, fast and handles non-linear interactions.
    - Extra Trees: Extremely randomized trees with random split thresholds, reducing ensemble variance.
    - Support Vector Classifier: Maximum margin RBF kernel classifier (requires scaling).
    
    Args:
        random_state: Seed for reproducibility.
        
    Returns:
        Dictionary mapping model names to scikit-learn estimators/pipelines.
    """
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=random_state,
                solver="lbfgs",
            ))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=15,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.08,
            max_leaf_nodes=31,
            min_samples_leaf=15,
            class_weight="balanced",
            random_state=random_state,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=150,
            max_depth=15,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "Support Vector Classifier": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel="rbf",
                C=2.0,
                gamma="scale",
                class_weight="balanced",
                random_state=random_state,
            ))
        ]),
    }
    return models


def evaluate_single_model_cv(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    cv: Optional[StratifiedKFold] = None,
    target_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Evaluates a single model using Stratified K-Fold cross validation.
    
    Computes fold-by-fold performance metrics, timing, out-of-fold predictions,
    per-class classification metrics, and aggregate confusion matrix.
    
    Args:
        model: Scikit-learn estimator or pipeline.
        X: Feature matrix (DataFrame).
        y: Target series.
        cv: StratifiedKFold instance (default 5 splits, shuffle=True, random_state=42).
        target_names: Class labels in order.
        
    Returns:
        Dictionary containing metric distributions, means, stds, reports, and confusion matrix.
    """
    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    if target_names is None:
        target_names = TARGET_CLASSES

    # Ensure y is 1D series
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]

    fold_metrics = {
        "macro_f1": [],
        "weighted_f1": [],
        "balanced_acc": [],
        "macro_precision": [],
        "macro_recall": [],
        "weighted_precision": [],
        "weighted_recall": [],
        "accuracy": [],
        "fit_time": [],
    }

    oof_preds = np.empty(len(y), dtype=object)

    # Manual fold iteration to ensure exact metric capture and fold tracking
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        t0 = time.time()
        from sklearn.base import clone
        fold_model = clone(model)
        fold_model.fit(X_tr, y_tr)
        fit_t = time.time() - t0

        y_pred = fold_model.predict(X_val)
        oof_preds[val_idx] = y_pred

        fold_metrics["macro_f1"].append(f1_score(y_val, y_pred, average="macro", zero_division=0))
        fold_metrics["weighted_f1"].append(f1_score(y_val, y_pred, average="weighted", zero_division=0))
        fold_metrics["balanced_acc"].append(balanced_accuracy_score(y_val, y_pred))
        fold_metrics["macro_precision"].append(precision_score(y_val, y_pred, average="macro", zero_division=0))
        fold_metrics["macro_recall"].append(recall_score(y_val, y_pred, average="macro", zero_division=0))
        fold_metrics["weighted_precision"].append(precision_score(y_val, y_pred, average="weighted", zero_division=0))
        fold_metrics["weighted_recall"].append(recall_score(y_val, y_pred, average="weighted", zero_division=0))
        fold_metrics["accuracy"].append(accuracy_score(y_val, y_pred))
        fold_metrics["fit_time"].append(fit_t)

    # Out-of-Fold (OOF) confusion matrix and classification report
    cm = confusion_matrix(y, oof_preds, labels=target_names)
    report_dict = classification_report(
        y, oof_preds, labels=target_names, target_names=target_names, output_dict=True, zero_division=0
    )

    summary = {
        "macro_f1_mean": float(np.mean(fold_metrics["macro_f1"])),
        "macro_f1_std": float(np.std(fold_metrics["macro_f1"])),
        "weighted_f1_mean": float(np.mean(fold_metrics["weighted_f1"])),
        "weighted_f1_std": float(np.std(fold_metrics["weighted_f1"])),
        "balanced_acc_mean": float(np.mean(fold_metrics["balanced_acc"])),
        "balanced_acc_std": float(np.std(fold_metrics["balanced_acc"])),
        "macro_precision_mean": float(np.mean(fold_metrics["macro_precision"])),
        "macro_precision_std": float(np.std(fold_metrics["macro_precision"])),
        "macro_recall_mean": float(np.mean(fold_metrics["macro_recall"])),
        "macro_recall_std": float(np.std(fold_metrics["macro_recall"])),
        "weighted_precision_mean": float(np.mean(fold_metrics["weighted_precision"])),
        "weighted_recall_mean": float(np.mean(fold_metrics["weighted_recall"])),
        "accuracy_mean": float(np.mean(fold_metrics["accuracy"])),
        "fit_time_mean": float(np.mean(fold_metrics["fit_time"])),
        "fold_metrics": fold_metrics,
        "oof_predictions": oof_preds,
        "confusion_matrix": cm,
        "classification_report": report_dict,
    }
    return summary


def evaluate_all_baseline_models(
    models: Optional[Dict[str, Any]] = None,
    X: Optional[pd.DataFrame] = None,
    y: Optional[pd.Series] = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Runs systematic cross-validation on all baseline models and compiles a comparative leaderboard.
    
    Args:
        models: Dictionary of models (default uses get_baseline_models()).
        X: Processed training features.
        y: Training target labels.
        random_state: Seed for cross-validation splitting.
        
    Returns:
        Tuple of (leaderboard_df, detailed_results_dict).
    """
    if models is None:
        models = get_baseline_models(random_state=random_state)
    if X is None:
        X = pd.read_csv("data/processed/X_train_processed.csv")
    if y is None:
        y_df = pd.read_csv("data/processed/y_train.csv")
        y = y_df.iloc[:, 0]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    results = {}
    records = []

    for name, model in models.items():
        print(f"--> Evaluating {name} across 5 Stratified Folds...")
        eval_res = evaluate_single_model_cv(model, X, y, cv=cv, target_names=TARGET_CLASSES)
        results[name] = eval_res

        records.append({
            "Model": name,
            "Macro F1": f"{eval_res['macro_f1_mean']:.4f} ± {eval_res['macro_f1_std']:.4f}",
            "Macro F1 (Raw)": eval_res["macro_f1_mean"],
            "Macro Recall": f"{eval_res['macro_recall_mean']:.4f} ± {eval_res['macro_recall_std']:.4f}",
            "Macro Recall (Raw)": eval_res["macro_recall_mean"],
            "Balanced Accuracy": f"{eval_res['balanced_acc_mean']:.4f} ± {eval_res['balanced_acc_std']:.4f}",
            "Balanced Accuracy (Raw)": eval_res["balanced_acc_mean"],
            "Weighted F1": f"{eval_res['weighted_f1_mean']:.4f} ± {eval_res['weighted_f1_std']:.4f}",
            "Macro Precision": f"{eval_res['macro_precision_mean']:.4f} ± {eval_res['macro_precision_std']:.4f}",
            "Accuracy": f"{eval_res['accuracy_mean']:.4f}",
            "Fit Time (s)": f"{eval_res['fit_time_mean']:.2f}s",
            "Imbalance Strategy": "class_weight='balanced'",
            "Validation Strategy": "Stratified 5-Fold CV",
        })

    leaderboard_df = pd.DataFrame(records).sort_values(by="Macro F1 (Raw)", ascending=False).reset_index(drop=True)
    return leaderboard_df, results
