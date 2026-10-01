"""
Model Tuning, Optimization, and Feature Exploration Engine
Course: IT3051 - Fundamentals of Data Mining (SLIIT)
Project: Predictive Maintenance Diagnostic System (AI4I-PMDI Dataset)
Group: 05 - 'Cognita'

This module conducts systematic empirical experiments for Stage 7:
1. Feature Engineering Ablation (Experiment A: Raw vs Experiment B: Raw + Domain Physics)
2. Imbalance Handling Strategies (None vs class_weight='balanced' vs SMOTE)
3. Feature Selection Investigation (Permutation Importance thresholding)
4. Hyperparameter Optimization (RandomizedSearchCV with StratifiedKFold on Macro F1)
"""

import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import f1_score, recall_score, balanced_accuracy_score, precision_score
from sklearn.inspection import permutation_importance
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from src.model_training import evaluate_single_model_cv, TARGET_CLASSES

# Domain Physics features engineered in Stage 4
PHYSICS_COLS = [
    "num__Temp_Difference",
    "num__Mechanical_Power_W",
    "num__Overstrain_Product",
    "num__Missing_Sensors_Count",
    "num__missingindicator_Temp_Difference",
    "num__missingindicator_Mechanical_Power_W",
    "num__missingindicator_Overstrain_Product",
]


def run_feature_ablation_experiment(
    X: pd.DataFrame,
    y: pd.Series,
    cv: Optional[StratifiedKFold] = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Compares Experiment A (Raw features) vs Experiment B (Raw + Domain Physics features).
    
    Demonstrates empirically whether domain-specific continuous physical interactions
    (Delta T, Mechanical Power, Overstrain Product) improve Macro-F1 and minority recall.
    
    Args:
        X: Full processed feature matrix (23 features).
        y: Target series.
        cv: Cross-validation splitter.
        random_state: Random state seed.
        
    Returns:
        Tuple of (comparison_df, raw_results_dict).
    """
    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    raw_feature_cols = [c for c in X.columns if c not in PHYSICS_COLS]
    full_feature_cols = list(X.columns)

    X_raw = X[raw_feature_cols]
    X_full = X[full_feature_cols]

    models_to_test = {
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=15, min_samples_split=4, min_samples_leaf=2,
            class_weight="balanced", random_state=random_state, n_jobs=-1
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=15,
            class_weight="balanced", random_state=random_state
        ),
    }

    records = []
    results = {}

    for model_name, clf in models_to_test.items():
        print(f"--> [Ablation] Testing {model_name} on Experiment A (Raw Features: {len(raw_feature_cols)} cols)...")
        res_a = evaluate_single_model_cv(clf, X_raw, y, cv=cv, target_names=TARGET_CLASSES)
        results[f"{model_name}_ExpA_Raw"] = res_a

        print(f"--> [Ablation] Testing {model_name} on Experiment B (Raw + Physics Features: {len(full_feature_cols)} cols)...")
        res_b = evaluate_single_model_cv(clf, X_full, y, cv=cv, target_names=TARGET_CLASSES)
        results[f"{model_name}_ExpB_Full"] = res_b

        macro_f1_diff = res_b["macro_f1_mean"] - res_a["macro_f1_mean"]
        macro_rec_diff = res_b["macro_recall_mean"] - res_a["macro_recall_mean"]
        bal_acc_diff = res_b["balanced_acc_mean"] - res_a["balanced_acc_mean"]

        records.append({
            "Model": model_name,
            "Exp A (Raw) Macro F1": f"{res_a['macro_f1_mean']:.4f} ± {res_a['macro_f1_std']:.4f}",
            "Exp B (Physics) Macro F1": f"{res_b['macro_f1_mean']:.4f} ± {res_b['macro_f1_std']:.4f}",
            "Macro F1 Lift": f"{macro_f1_diff:+.4f} ({macro_f1_diff/res_a['macro_f1_mean']*100:+.2f}%)",
            "Exp A Macro Recall": f"{res_a['macro_recall_mean']:.4f}",
            "Exp B Macro Recall": f"{res_b['macro_recall_mean']:.4f}",
            "Macro Recall Lift": f"{macro_rec_diff:+.4f}",
            "Exp A Balanced Acc": f"{res_a['balanced_acc_mean']:.4f}",
            "Exp B Balanced Acc": f"{res_b['balanced_acc_mean']:.4f}",
            "Balanced Acc Lift": f"{bal_acc_diff:+.4f}",
            "Physics Lift Significant": "YES" if macro_f1_diff > 0 else "NO",
        })

    ablation_df = pd.DataFrame(records)
    return ablation_df, results


def run_imbalance_strategy_experiment(
    X: pd.DataFrame,
    y: pd.Series,
    cv: Optional[StratifiedKFold] = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Compares three class imbalance strategies inside CV folds:
    1. Unweighted (Standard baseline)
    2. Algorithmic Cost-Weighting (class_weight='balanced')
    3. Synthetic Minority Oversampling (SMOTE with k_neighbors=2 inside pipeline)
    
    Args:
        X: Processed features.
        y: Target series.
        cv: Cross-validation splitter.
        random_state: Random state seed.
        
    Returns:
        Tuple of (strategy_comparison_df, raw_results_dict).
    """
    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # Imbalance strategies benchmarked on Random Forest
    strategies = {
        "Unweighted Baseline": RandomForestClassifier(
            n_estimators=150, max_depth=15, min_samples_split=4, min_samples_leaf=2,
            class_weight=None, random_state=random_state, n_jobs=-1
        ),
        "Class Weighting ('balanced')": RandomForestClassifier(
            n_estimators=150, max_depth=15, min_samples_split=4, min_samples_leaf=2,
            class_weight="balanced", random_state=random_state, n_jobs=-1
        ),
        "SMOTE (k=2) inside CV": ImbPipeline([
            ("smote", SMOTE(k_neighbors=2, random_state=random_state)),
            ("clf", RandomForestClassifier(
                n_estimators=150, max_depth=15, min_samples_split=4, min_samples_leaf=2,
                class_weight=None, random_state=random_state, n_jobs=-1
            ))
        ]),
    }

    records = []
    results = {}

    for strat_name, pipeline in strategies.items():
        print(f"--> [Imbalance] Testing Strategy: {strat_name}...")
        res = evaluate_single_model_cv(pipeline, X, y, cv=cv, target_names=TARGET_CLASSES)
        results[strat_name] = res

        records.append({
            "Imbalance Strategy": strat_name,
            "Macro F1": f"{res['macro_f1_mean']:.4f} ± {res['macro_f1_std']:.4f}",
            "Macro F1 (Raw)": res["macro_f1_mean"],
            "Macro Recall": f"{res['macro_recall_mean']:.4f} ± {res['macro_recall_std']:.4f}",
            "Macro Recall (Raw)": res["macro_recall_mean"],
            "Balanced Accuracy": f"{res['balanced_acc_mean']:.4f} ± {res['balanced_acc_std']:.4f}",
            "Weighted F1": f"{res['weighted_f1_mean']:.4f}",
            "Macro Precision": f"{res['macro_precision_mean']:.4f}",
            "Fit Time (s)": f"{res['fit_time_mean']:.2f}s",
        })

    imbalance_df = pd.DataFrame(records).sort_values(by="Macro F1 (Raw)", ascending=False).reset_index(drop=True)
    return imbalance_df, results


def run_feature_selection_investigation(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    cv: Optional[StratifiedKFold] = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Computes Permutation Importance on training folds and tests feature selection subsets.
    
    Evaluates:
    - Full Feature Set (23 features)
    - Top 15 Features (retaining highest importance)
    - Top 10 Features
    
    Args:
        model: Fitted baseline estimator.
        X: Processed features.
        y: Target series.
        cv: Cross-validation splitter.
        random_state: Random state seed.
        
    Returns:
        Tuple of (importance_df, subset_performance_df).
    """
    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # Fit a single model on all X_train to compute permutation importance
    from sklearn.base import clone
    clf = clone(model)
    clf.fit(X, y)

    perm_res = permutation_importance(
        clf, X, y, scoring="f1_macro", n_repeats=5, random_state=random_state, n_jobs=-1
    )

    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance Mean": perm_res.importances_mean,
        "Importance Std": perm_res.importances_std,
    }).sort_values(by="Importance Mean", ascending=False).reset_index(drop=True)

    # Benchmark feature subsets
    top_15_cols = list(importance_df["Feature"].iloc[:15])
    top_10_cols = list(importance_df["Feature"].iloc[:10])

    subsets = {
        "Full Feature Set (23 features)": X,
        "Top 15 Features": X[top_15_cols],
        "Top 10 Features": X[top_10_cols],
    }

    subset_records = []
    for subset_name, X_sub in subsets.items():
        print(f"--> [Feature Selection] Evaluating subset: {subset_name}...")
        res = evaluate_single_model_cv(model, X_sub, y, cv=cv, target_names=TARGET_CLASSES)
        subset_records.append({
            "Feature Configuration": subset_name,
            "Feature Count": X_sub.shape[1],
            "Macro F1": f"{res['macro_f1_mean']:.4f} ± {res['macro_f1_std']:.4f}",
            "Macro F1 (Raw)": res["macro_f1_mean"],
            "Macro Recall": f"{res['macro_recall_mean']:.4f} ± {res['macro_recall_std']:.4f}",
            "Balanced Accuracy": f"{res['balanced_acc_mean']:.4f} ± {res['balanced_acc_std']:.4f}",
            "Weighted F1": f"{res['weighted_f1_mean']:.4f}",
        })

    subset_df = pd.DataFrame(subset_records).sort_values(by="Macro F1 (Raw)", ascending=False).reset_index(drop=True)
    return importance_df, subset_df


def tune_hyperparameters(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    cv: Optional[StratifiedKFold] = None,
    n_iter: int = 20,
    random_state: int = 42,
) -> Tuple[Any, Dict[str, Any], pd.DataFrame]:
    """Tunes hyperparameters for top models using RandomizedSearchCV with StratifiedKFold on Macro F1.
    
    Args:
        model_name: "Random Forest" or "HistGradientBoosting".
        X: Processed features.
        y: Target series.
        cv: Cross-validation splitter.
        n_iter: Number of parameter combinations to sample.
        random_state: Seed for search.
        
    Returns:
        Tuple of (best_estimator, best_params, cv_results_df).
    """
    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    if model_name == "Random Forest":
        base_model = RandomForestClassifier(random_state=random_state, n_jobs=-1)
        param_dist = {
            "n_estimators": [100, 150, 200, 250],
            "max_depth": [10, 15, 20, 25, None],
            "min_samples_split": [2, 4, 6, 8],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", 0.6, 0.8],
            "class_weight": ["balanced", "balanced_subsample"],
        }
    elif model_name == "HistGradientBoosting":
        base_model = HistGradientBoostingClassifier(random_state=random_state)
        param_dist = {
            "max_iter": [100, 150, 200, 250],
            "learning_rate": [0.03, 0.05, 0.08, 0.1, 0.15],
            "max_leaf_nodes": [15, 31, 45, 63],
            "min_samples_leaf": [10, 15, 20, 30],
            "l2_regularization": [0.0, 0.1, 1.0, 5.0],
            "class_weight": ["balanced"],
        }
    else:
        raise ValueError(f"Tuning not configured for model: {model_name}")

    print(f"--> [Tuning] Launching RandomizedSearchCV on {model_name} (n_iter={n_iter}, scoring='f1_macro')...")
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="f1_macro",
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    search.fit(X, y)

    cv_results_df = pd.DataFrame(search.cv_results_).sort_values(by="rank_test_score").reset_index(drop=True)
    print(f"--> [Tuning] Best Macro F1 for {model_name}: {search.best_score_:.4f}")
    print(f"--> [Tuning] Best Parameters: {search.best_params_}")

    return search.best_estimator_, search.best_params_, cv_results_df
