# IT3051 - Fundamentals of Data Mining Mini Project
## Predictive Maintenance Diagnostic System (AI4I-PMDI Dataset)

**Institution:** Sri Lanka Institute of Information Technology (SLIIT)  
**Course:** IT3051 - Fundamentals of Data Mining (Year 3 Semester 2)  
**Group:** 05 - "Cognita"  
**Dataset:** AI4I-PMDI Predictive Maintenance Dataset (10,000 instances, 12 features)  
**Problem:** Multiclass Machine Health Diagnostic Classification under Severe Class Imbalance and Realistic Sensor Missingness.

---

## 📌 Project Overview

This repository contains the complete end-to-end data mining project for predictive maintenance of industrial CNC milling machinery. The goal is to classify machine operating states into 6 diagnostic conditions:
1. `No failure` (Normal Operation)
2. `Heat Dissipation Failure (HDF)`
3. `Power Failure (PWF)`
4. `Overstrain Failure (OSF)`
5. `Tool Wear Failure (TWF)`
6. `Random Failures (RNF)`

The project addresses realistic industrial challenges including severe class imbalance ($508 : 1$ ratio between majority and rarest class), realistic sensor missingness ($33\% - 67\%$ across sensor channels), and builds an exportable Scikit-learn preprocessing pipeline for real-time web application inference.

---

## 🗂️ Repository Structure

```
.
├── data/
│   ├── raw/
│   │   └── AI4I-_PMDI_-_Maintenance_dataset.csv  # Raw dataset with realistic missing values
│   └── processed/
│       ├── train.csv                             # Cleaned 80% training set (8,000 instances)
│       ├── test.csv                              # Cleaned 20% test set (2,000 instances)
│       └── split_indices.json                    # Exact reproducible split indices & distributions
├── notebooks/
│   ├── 01_eda.ipynb                              # Stage 3: Full Exploratory Data Analysis
│   └── 02_preprocessing_feature_engineering.ipynb# Stage 4: Preprocessing & Feature Engineering
├── src/
│   ├── __init__.py
│   ├── eda_utils.py                              # Modular data audit, hypothesis tests & physics rules
│   ├── features.py                               # Custom PhysicsFeatureEngineer transformer
│   ├── preprocessing.py                          # Splitting, imputation CV, pipeline & assertions
│   ├── generate_figures.py                       # Standalone high-res figure generation script
│   ├── build_notebook.py                         # EDA notebook compiler
│   ├── build_preprocessing_notebook.py           # Preprocessing notebook compiler
│   └── run_preprocessing_stage.py               # Preprocessing pipeline runner & test assertions
├── reports/
│   ├── eda_findings.md                           # Empirical findings & proposal verification matrix
│   ├── decision_log.md                           # Evidence-backed decision log for viva defense
│   ├── processed_data_summary.md                 # Summary of training and test splits
│   └── figures/                                  # 10 High-resolution publication plots (.png)
├── models/
│   └── preprocessor.joblib                       # Fitted scikit-learn Pipeline for inference
├── Dataset proposal - 05_Cognita.pdf             # Group 05 official project proposal document
└── README.md
```

---

## 📊 Summary of Completed Stages

### Stage 1 & 2: Problem Formulation & Dataset Selection
- Selection and validation of the AI4I-PMDI predictive maintenance dataset.
- Formal project proposal document: [`Dataset proposal - 05_Cognita.pdf`](./Dataset%20proposal%20-%2005_Cognita.pdf).

### Stage 3: Exploratory Data Analysis (EDA)
- **$100\%$ Proposal Verification**: All $10,000$ instances, $12$ attributes, target proportions, and sensor missing rates verified.
- **Missingness Mechanism (MAR)**: Proved missingness is completely governed by the `Control` multiplexer mode ($A, B, C$).
- **Outlier Failure Concentration**: Proved **$92.2\%$ of Torque IQR outliers are actual breakdown events** ($50$ PWF, $9$ OSF), justifying zero outlier trimming.
- **Domain Physics Validation**: Validated AI4I physical failure rules (HDF, PWF, OSF, TWF) hold with **$100\%$ precision & recall** on monitored channels.
- **Full Notebook**: [`notebooks/01_eda.ipynb`](./notebooks/01_eda.ipynb).
- **Findings Report**: [`reports/eda_findings.md`](./reports/eda_findings.md).

### Stage 4: Preprocessing & Feature Engineering
- **Strict Stratified 80/20 Split**: Locked before any fitting (`train.csv` / `test.csv`).
- **Imputation Benchmarking**: Evaluated Median, KNN, IterativeImputer with/without `MissingIndicator` via Stratified 5-Fold CV. `Median + MissingIndicator` won decisively (**Macro-F1: $0.6785 \pm 0.0153$**).
- **Domain Physics Feature Engineering**: Custom `PhysicsFeatureEngineer` transformer deriving $\Delta T$, Mechanical Power ($P$), Overstrain Product, and Missing Sensors Count.
- **Feature Importance**: Permutation importance confirmed engineered physics features occupy the **top 5 most important predictors** (`Mechanical_Power_W`: $0.1920$, `Overstrain_Product`: $0.1721$, `Temp_Difference`: $0.1493$).
- **Exportable Pipeline**: Fitted and serialized [`models/preprocessor.joblib`](./models/preprocessor.joblib) capable of handling single-row inputs with `NaN` fields at inference.
- **Full Notebook**: [`notebooks/02_preprocessing_feature_engineering.ipynb`](./notebooks/02_preprocessing_feature_engineering.ipynb).
- **Decision Log**: [`reports/decision_log.md`](./reports/decision_log.md).


### Stage 6: Baseline Model Development
- **5 Diverse Algorithms Benchmarked:** Multinomial Logistic Regression, Support Vector Classifier (RBF), Random Forest, Extra Trees, and HistGradientBoosting.
- **Stratified 5-Fold Cross-Validation:** $N=8,000$ training partition with preserved minority representation (3 instances of Random Failures per fold).
- **Leak-Free Protocol:** Untouched 2,000-instance test set locked throughout model development.
- **Metric Evaluation:** Prioritized **Macro-F1** and **Balanced Accuracy/Macro Recall** over misleading accuracy.
- **Baseline Leaderboard:** HistGradientBoosting ($0.6830$ Macro-F1) and Random Forest ($0.6744$ Macro-F1, $0.7828$ Recall) led the benchmarks.

### Stage 7: Model Optimization & Final Model Selection
- **Feature Engineering Ablation:** Proved continuous domain physics features ($\Delta T$, Spindle Power $W$, Overstrain Product) deliver a **$+14.89\%$ relative lift in Macro-F1** on Random Forest ($0.5870 \rightarrow 0.6744$).
- **Imbalance Handling Strategy:** Algorithmic cost-weighting (`class_weight='balanced_subsample'`) outperformed SMOTE, providing higher recall ($78.3\%$ vs $69.0\%$) and $3\times$ lower cross-fold variance without synthetic artifact noise.
- **Systematic Hyperparameter Tuning:** `RandomizedSearchCV` on Macro-F1 lifted Tuned Random Forest to **$0.7000 \pm 0.0101$ Macro-F1**.
- **Champion Model Selected:** Tuned Random Forest (`n_estimators=200`, `max_depth=10`, `class_weight='balanced_subsample'`).
- **Untouched Test Set Evaluation ($N=2,000$):**
  - **Accuracy:** $98.50\%$
  - **Macro-F1 Score:** **$0.7101$**
  - **Balanced Accuracy / Macro Recall:** **$0.7606$**
  - **Heat Dissipation Failure (HDF):** Precision $1.00$, Recall $1.00$, F1 **$1.00$**
  - **Power Failure (PWF):** Precision $1.00$, Recall $1.00$, F1 **$1.00$**
  - **Overstrain Failure (OSF):** Precision $1.00$, Recall $0.95$, F1 **$0.97$**
- **Production Pipeline Export:** Fully serialized end-to-end [`models/champion_pipeline.joblib`](./models/champion_pipeline.joblib) capable of sub-millisecond ($0.4$ ms) inference with missing sensors.
- **Full Modeling Notebook:** [`notebooks/03_model_development_and_optimization.ipynb`](./notebooks/03_model_development_and_optimization.ipynb).
- **Comprehensive Report:** [`reports/model_development_and_optimization.md`](./reports/model_development_and_optimization.md).


---

## 🚀 Getting Started & Execution

### Prerequisites
```bash
python -m pip install -r requirements.txt
```

### Running the Entire Modeling & Optimization Pipeline
```bash
python src/run_modeling_stage.py
```

### Compiling and Executing the Complete Stage 6/7 Notebook
```bash
python src/build_modeling_notebook.py
```

### Launching the Interactive Web Application & API (Stages 9 & 10)
```bash
python -m uvicorn backend.main:app --reload --port 8000
```
Once started, access:
* **Interactive Web Dashboard:** `http://127.0.0.1:8000/`
* **Interactive OpenAPI Swagger Docs:** `http://127.0.0.1:8000/docs`
* **ReDoc API Documentation:** `http://127.0.0.1:8000/redoc`

### Running Backend Unit & Live End-to-End Tests
```bash
# Automated backend tests (12 tests)
pytest -v tests/test_backend.py

# Live end-to-end integration tests (all presets & held-out test data)
python tests/test_live_system.py
```

---

## 👥 Authors
- **Group 05 ("Cognita")**  
- Sri Lanka Institute of Information Technology (SLIIT)  
- IT3051 - Fundamentals of Data Mining (2026)



