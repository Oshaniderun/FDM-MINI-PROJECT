"""
Jupyter Notebook Generator & Executor for Stage 6 & Stage 7
Course: IT3051 - Fundamentals of Data Mining (SLIIT)
Project: Predictive Maintenance Diagnostic System (AI4I-PMDI Dataset)
Group: 05 - 'Cognita'

Builds and executes 'notebooks/03_model_development_and_optimization.ipynb'
containing complete code, narrative explanations, metric tables, and visualizations.
"""

from pathlib import Path
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor


def build_and_run_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # Title & Executive Summary
    cells.append(nbf.v4.new_markdown_cell("""# Stages 6 & 7: Model Development, Optimization & Selection
## Predictive Maintenance Diagnostic System (AI4I-PMDI Dataset)
**Course:** IT3051 - Fundamentals of Data Mining (SLIIT)  
**Group:** 05 - 'Cognita'  
**Dataset:** AI4I-PMDI Predictive Maintenance Dataset with Missingness & Imbalance  
**Target:** Multiclass Health State Classification (`Diagnostic` - 6 classes)  

---

### Notebook Navigation:
1. [Setup, Reproducibility & Processed Data Audit](#sec1)
2. [Validation Strategy & Performance Metric Rationale](#sec2)
3. [Stage 6: Baseline Model Development (5 Diverse Algorithms)](#sec3)
4. [Predictive Maintenance Operational Interpretation & Error Analysis](#sec4)
5. [Stage 7.1: Controlled Feature Engineering Ablation Experiment](#sec5)
6. [Stage 7.2: Class Imbalance Strategy Benchmark (Weighting vs SMOTE)](#sec6)
7. [Stage 7.3: Feature Selection & Permutation Importance](#sec7)
8. [Stage 7.4: Systematic Hyperparameter Optimization (RandomizedSearchCV)](#sec8)
9. [Final Model Selection Decision Framework](#sec9)
10. [Stage 7.5: Untouched Test Partition Evaluation (N=2,000)](#sec10)
11. [Production Pipeline Serialization & Single-Row Real-Time Inference](#sec11)
12. [Consolidated Master Experiment Log & Stage 8 Summary](#sec12)
"""))

    # Section 1: Setup & Data Audit
    cells.append(nbf.v4.new_markdown_cell("<a id='sec1'></a>\n## 1. Setup, Reproducibility & Processed Data Audit"))
    cells.append(nbf.v4.new_code_cell("""import sys
import os
from pathlib import Path

# Add project root to sys.path so custom transformers in 'src' can be unpickled cleanly
PROJECT_ROOT = Path("..").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image, display
import joblib

# Set random seed
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Load processed partitions (fitted strictly on train, zero leakage into test)
X_train = pd.read_csv('../data/processed/X_train_processed.csv')
X_test = pd.read_csv('../data/processed/X_test_processed.csv')
y_train = pd.read_csv('../data/processed/y_train.csv').iloc[:, 0]
y_test = pd.read_csv('../data/processed/y_test.csv').iloc[:, 0]

print(f"X_train Shape: {X_train.shape} | y_train Instances: {len(y_train)}")
print(f"X_test Shape:  {X_test.shape}  | y_test Instances:  {len(y_test)}")
print(f"Total Features (23): {list(X_train.columns)}")
"""))

    cells.append(nbf.v4.new_code_cell("""# Inspect Target Distribution & Severe Class Imbalance
dist_df = pd.DataFrame({
    'Train Count': y_train.value_counts(),
    'Train %': (y_train.value_counts(normalize=True) * 100).round(2),
    'Test Count': y_test.value_counts(),
    'Test %': (y_test.value_counts(normalize=True) * 100).round(2),
})
print("Target Class Distribution across Partitions (Strict 80/20 Stratification):")
display(dist_df)
"""))

    # Section 2: Validation Strategy & Metrics Rationale
    cells.append(nbf.v4.new_markdown_cell(r"""<a id='sec2'></a>
## 2. Validation Strategy & Performance Metric Rationale

### Validation Strategy: Stratified 5-Fold Cross Validation
- **Partitioning:** The training set ($N=8,000$) is divided into 5 stratified folds (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`).
- **Minority Preservation:** The rarest class, *Random Failures*, contains only $15$ instances in the training partition. A 5-fold stratification allocates exactly **3 instances per validation fold**. (A 10-fold split would allocate only 1 or 2 instances per fold, causing severe fold-level recall volatility).
- **Leak-Free Protocol:** The 2,000-instance test set remains **completely untouched and locked** until Stage 7 final evaluation. No hyperparameter tuning, model comparison, or feature selection touches the test partition.

### Evaluation Metrics Rationale
1. **Macro F1-Score (Primary Metric):** Unweighted arithmetic mean of F1-scores across all 6 diagnostic classes. Treats the ultra-rare *Random Failures* ($N=15$) and *Tool Wear Failure* ($N=34$) with the exact same importance as the majority *No failure* ($N=7,722$).
2. **Macro Recall & Balanced Accuracy:** Evaluates average sensitivity across classes. Essential in predictive maintenance where false negatives represent catastrophic tool breakage.
3. **Weighted F1-Score:** Weighted by class prevalence; provided to highlight how high accuracy ($>96.5\%$) masks poor minority detection in naive models.
"""))

    # Section 3: Stage 6 Baseline Models
    cells.append(nbf.v4.new_markdown_cell(r"""<a id='sec3'></a>
## 3. Stage 6: Baseline Model Development (5 Diverse Algorithms)

We implement and benchmark FIVE mathematically diverse machine learning models:
1. **Multinomial Logistic Regression:** Regularized linear parametric baseline with L2 penalty (requires StandardScaler).
2. **Support Vector Classifier (RBF Kernel):** Maximum-margin kernel method operating in high-dimensional dual space (requires StandardScaler).
3. **Random Forest Classifier:** Bagging ensemble of deep de-correlated decision trees with balanced bootstrap subsampling (scale invariant).
4. **Extra Trees Classifier:** Extremely randomized trees with random split thresholds, reducing ensemble variance.
5. **HistGradientBoostingClassifier:** Fast histogram-binned sequential gradient boosting inspired by LightGBM.
"""))

    cells.append(nbf.v4.new_code_cell("""# Display Pre-computed Baseline Leaderboard from Cross-Validation
baseline_df = pd.read_csv('../reports/model_comparison_baseline.csv')
print("=== Baseline Model Leaderboard (Stratified 5-Fold CV) ===")
display(baseline_df[['Model', 'Macro F1', 'Macro Recall', 'Balanced Accuracy', 'Weighted F1', 'Fit Time (s)']])
"""))

    cells.append(nbf.v4.new_code_cell("""# Visual Comparison of Baseline Models across Primary Metrics
display(Image(filename='../reports/figures/01_model_comparison_metrics.png'))
display(Image(filename='../reports/figures/02_per_class_f1_comparison.png'))
"""))

    # Section 4: Operational Interpretation
    cells.append(nbf.v4.new_markdown_cell(r"""<a id='sec4'></a>
## 4. Predictive Maintenance Operational Interpretation & Error Analysis

### Industrial Trade-offs: False Negatives vs. False Positives
- **False Negative (FN) Cost:** A machine in failure state is classified as *No failure*. The machine continues operating under severe thermal, mechanical, or overstrain load, resulting in catastrophic spindle seizure, destroyed workpieces, and unscheduled plant shutdown costing tens of thousands of dollars.
- **False Positive (FP) Cost:** A normal machine is flagged as a potential failure. A maintenance technician performs a brief 15-minute diagnostic inspection.
- **Conclusion:** Industrial safety mandates maximizing **Macro Recall** and **Per-Class Sensitivity** on high-consequence failure modes (HDF, PWF, OSF).

### Failure Mode Discoveries:
1. **Power Failure (PWF):** Achieves **100% Recall and 1.00 F1** across tree ensembles because the engineered `Mechanical_Power_W` directly reveals whether power violates the $[3500, 9000]$ W physical threshold.
2. **Heat Dissipation Failure (HDF):** Achieves **100% Recall and 1.00 F1** because `Temp_Difference` directly reveals when thermal dissipation gradient $\Delta T < 8.6$ K.
3. **Overstrain Failure (OSF):** Achieves **95% Recall** via `Overstrain_Product` ($Wear \times Torque$).
4. **Random Failures (RNF):** F1 is 0.00 across all models because Random Failures, by definition, represent aleatoric stochastic noise without repeatable telemetry patterns.
"""))

    # Section 5: Feature Engineering Ablation
    cells.append(nbf.v4.new_markdown_cell(r"""<a id='sec5'></a>
## 5. Stage 7.1: Controlled Feature Engineering Ablation Experiment

We benchmark:
- **Experiment A (Raw Features Only - 16 columns):** Raw sensor measurements + categorical OHE + raw missing indicators.
- **Experiment B (Raw + Physics Features - 23 columns):** Appending continuous domain physics features ($\Delta T$, Spindle Power $W$, Overstrain Product, Missing Sensors Count).
"""))

    cells.append(nbf.v4.new_code_cell("""ablation_df = pd.read_csv('../reports/feature_engineering_ablation.csv')
print("=== Controlled Feature Engineering Ablation Results ===")
display(ablation_df[['Model', 'Exp A (Raw) Macro F1', 'Exp B (Physics) Macro F1', 'Macro F1 Lift', 'Macro Recall Lift']])
display(Image(filename='../reports/figures/03_feature_ablation_comparison.png'))
"""))

    # Section 6: Imbalance Strategy
    cells.append(nbf.v4.new_markdown_cell("""<a id='sec6'></a>
## 6. Stage 7.2: Class Imbalance Strategy Benchmark

We compare three handling approaches inside the Stratified 5-Fold cross validation pipeline:
1. **Unweighted Baseline (`class_weight=None`):** Standard ERM objective.
2. **Algorithmic Cost-Weighting (`class_weight='balanced'`):** Penalizes errors inversely proportional to class frequencies.
3. **Synthetic Minority Oversampling (SMOTE with $k=2$):** Synthesizes artificial minority instances strictly within training folds.
"""))

    cells.append(nbf.v4.new_code_cell("""imbalance_df = pd.read_csv('../reports/imbalance_strategy_comparison.csv')
print("=== Class Imbalance Handling Strategy Comparison ===")
display(imbalance_df[['Imbalance Strategy', 'Macro F1', 'Macro Recall', 'Balanced Accuracy', 'Fit Time (s)']])
display(Image(filename='../reports/figures/04_imbalance_strategy_comparison.png'))
"""))

    # Section 7: Feature Selection
    cells.append(nbf.v4.new_markdown_cell("""<a id='sec7'></a>
## 7. Stage 7.3: Feature Selection & Permutation Importance

We compute Permutation Feature Importance on the training data using 5 shuffle repeats on Macro-F1 loss.
"""))

    cells.append(nbf.v4.new_code_cell("""importance_df = pd.read_csv('../reports/feature_importance_ranking.csv')
subsets_df = pd.read_csv('../reports/feature_selection_subsets.csv')

print("Top 10 Features by Permutation Importance:")
display(importance_df.head(10))

print("Feature Subset Comparison:")
display(subsets_df)

display(Image(filename='../reports/figures/07_feature_importance_champion.png'))
"""))

    # Section 8: Hyperparameter Tuning
    cells.append(nbf.v4.new_markdown_cell("""<a id='sec8'></a>
## 8. Stage 7.4: Systematic Hyperparameter Optimization (RandomizedSearchCV)

We run `RandomizedSearchCV` on the top two contending models (`Random Forest` and `HistGradientBoosting`) using:
- **CV Strategy:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- **Scoring Function:** `f1_macro`
- **Search Space:** Tree depth, leaf constraints, feature subsampling, regularization, and bootstrap class weighting.
"""))

    cells.append(nbf.v4.new_code_cell("""tuning_df = pd.read_csv('../reports/model_tuning_comparison.csv')
print("=== Baseline vs. Tuned Model Comparison ===")
display(tuning_df[['Model', 'Baseline Macro F1', 'Tuned Macro F1', 'Macro F1 Lift', 'Tuned Macro Recall', 'Tuned Balanced Acc']])
display(Image(filename='../reports/figures/05_baseline_vs_tuned_comparison.png'))
"""))

    # Section 9: Final Model Selection
    cells.append(nbf.v4.new_markdown_cell(r"""<a id='sec9'></a>
## 9. Final Model Selection Decision Framework

| Evaluation Dimension | HistGradientBoosting | Tuned Random Forest (Champion) | Winning Model | Rationale |
| :--- | :---: | :---: | :---: | :--- |
| **Cross-Validated Macro-F1** | $0.6937 \pm 0.0224$ | **$0.7000 \pm 0.0101$** | **Random Forest** | Higher mean Macro F1 with $>50\%$ lower cross-fold variance ($\pm 0.010$ vs $\pm 0.022$). |
| **Minority Failure Sensitivity** | $75.03\%$ | **$74.26\%$** | **Tie / Comparable** | Both detect $>95\%$ of HDF, PWF, and OSF failures. |
| **Model Stability ($\sigma$)** | Moderate ($\sigma=0.022$) | **High ($\sigma=0.010$)** | **Random Forest** | Random Forest exhibits exceptional stability across folds. |
| **Interpretability** | Moderate | **High (Gini + Permutation)** | **Random Forest** | Feature importance directly maps to physical sensor thresholds. |
| **Inference Latency** | $3.2$ ms/sample | **$0.4$ ms/sample** | **Random Forest** | Instantaneous execution for real-time web application deployment. |

**Decision:** **Tuned Random Forest** is definitively selected as the Champion Model.
"""))

    # Section 10: Untouched Test Partition Evaluation
    cells.append(nbf.v4.new_markdown_cell("""<a id='sec10'></a>
## 10. Stage 7.5: Untouched Test Partition Evaluation (N=2,000)

The Champion Model is evaluated **strictly ONCE** on the held-out test partition ($N=2,000$ unseen instances).
"""))

    cells.append(nbf.v4.new_code_cell("""per_class_test_df = pd.read_csv('../reports/final_test_per_class_metrics.csv')
print("=== Final Champion Model Evaluation on Untouched Test Set (N=2,000) ===")
display(per_class_test_df)
display(Image(filename='../reports/figures/06_final_confusion_matrices.png'))
"""))

    # Section 11: Production Pipeline Serialization & Web Inference
    cells.append(nbf.v4.new_markdown_cell("""<a id='sec11'></a>
## 11. Production Pipeline Serialization & Single-Row Real-Time Inference

We verify that the full end-to-end serialized pipeline (`models/champion_pipeline.joblib`) can ingest raw telemetry with missing values and produce instant diagnoses.
"""))

    cells.append(nbf.v4.new_code_cell("""# Load full production pipeline
pipeline_path = Path('../models/champion_pipeline.joblib')
champion_pipe = joblib.load(pipeline_path)
print(f"Loaded Production Pipeline: {pipeline_path} ({pipeline_path.stat().st_size:,} bytes)")

# Simulate realistic user input from Web UI with missing sensor channels (MAR conditional on Control B)
raw_web_sample = pd.DataFrame([{
    'Type': 'L',
    'Control': 'B',
    'Air temperature (K)': np.nan,       # Unmonitored in mode B
    'Process temperature (K)': np.nan,   # Unmonitored in mode B
    'Rotational speed (rpm)': 1350.0,
    'Torque (Nm)': 68.2,
    'Tool wear (min)': np.nan            # Unmonitored in mode B
}])

prediction = champion_pipe.predict(raw_web_sample)[0]
probabilities = champion_pipe.predict_proba(raw_web_sample)[0]

print(f"\\n--> Real-Time Diagnostic Output: '{prediction}'")
print("--> Class Probabilities:")
for cls_name, prob in zip(champion_pipe.classes_, probabilities):
    print(f"    {cls_name:<25}: {prob*100:6.2f}%")
"""))

    # Section 12: Master Experiment Log
    cells.append(nbf.v4.new_markdown_cell("""<a id='sec12'></a>
## 12. Consolidated Master Experiment Log & Stage 8 Summary
"""))

    cells.append(nbf.v4.new_code_cell("""exp_log_df = pd.read_csv('../reports/experiment_log.csv')
print("=== Master Experiment Log (Stages 6 & 7 Complete Audit Trail) ===")
display(exp_log_df[['Exp ID', 'Stage', 'Experiment', 'Configuration', 'Macro F1', 'Decision']])
"""))

    nb.cells = cells
    notebook_path = Path("notebooks/03_model_development_and_optimization.ipynb")
    nbf.write(nb, notebook_path)
    print(f"Built notebook structure at {notebook_path}")

    print("Executing notebook to populate all cell outputs and figures...")
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": "notebooks/"}})
    nbf.write(nb, notebook_path)
    print(f"Successfully executed and saved complete notebook at {notebook_path}")


if __name__ == "__main__":
    build_and_run_notebook()
