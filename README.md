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
├── docs/
│   └── viva_prep.md                              # 15+ Viva Q&As with empirical metrics & justifications
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
- **Viva Defense Guide**: [`docs/viva_prep.md`](./docs/viva_prep.md).

---

## 🚀 Getting Started

### Prerequisites
```bash
python -m pip install -r requirements.txt
```
Or install core packages:
```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn imbalanced-learn joblib nbformat ipykernel
```

### Running the Preprocessing Pipeline & Sanity Tests
```bash
python src/run_preprocessing_stage.py
```

### Generating All EDA Figures
```bash
python src/generate_figures.py
```

---

## 👥 Authors
- **Group 05 ("Cognita")**  
- Sri Lanka Institute of Information Technology (SLIIT)  
- IT3051 - Fundamentals of Data Mining (2026)
