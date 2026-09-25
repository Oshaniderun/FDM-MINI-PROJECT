# Exploratory Data Analysis (EDA) Findings Report
**Course:** IT3051 - Fundamentals of Data Mining (SLIIT)  
**Project:** Machine Predictive Maintenance Diagnostic System  
**Group:** 05 - "Cognita"  
**Dataset:** AI4I-PMDI (AI4I Predictive Maintenance Dataset with Missing Values)  
**Deliverable Scope:** Stage 3 - Comprehensive EDA Findings & Empirical Verifications  

---

## 1. Executive Summary & Proposal Verification

An empirical audit was conducted on the raw AI4I-PMDI dataset (`AI4I- PMDI - Maintenance dataset.csv`). Every metric reported in the project proposal was verified against the data.

### Verification Matrix (Proposal vs. Empirical Findings)

| Metric / Dimension | Proposal Reported Value | Verified Empirical Value | Discrepancy / Status |
| :--- | :--- | :--- | :--- |
| **Total Rows ($N$)** | 10,000 | 10,000 | Exact Match (0% error) |
| **Total Columns ($P$)** | 12 | 12 | Exact Match |
| **Air temperature (K) Missing %** | 65.63% | 65.63% (6,563 rows) | Exact Match |
| **Process temperature (K) Missing %**| 65.63% | 65.63% (6,563 rows) | Exact Match |
| **Rotational speed (rpm) Missing %** | 33.21% | 33.21% (3,321 rows) | Exact Match |
| **Torque (Nm) Missing %** | 34.37% | 34.37% (3,437 rows) | Exact Match |
| **Tool wear (min) Missing %** | 66.79% | 66.79% (6,679 rows) | Exact Match |
| **Non-Sensor Missing %** | 0.00% | 0.00% (0 rows) | Exact Match |
| **Target: No failure** | 9,652 (96.52%) | 9,652 (96.52%) | Exact Match |
| **Target: Heat Dissipation Failure** | 106 (1.06%) | 106 (1.06%) | Exact Match |
| **Target: Overstrain Failure** | 98 (0.98%) | 98 (0.98%) | Exact Match |
| **Target: Power Failure** | 83 (0.83%) | 83 (0.83%) | Exact Match |
| **Target: Tool Wear Failure** | 42 (0.42%) | 42 (0.42%) | Exact Match |
| **Target: Random Failures** | 19 (0.19%) | 19 (0.19%) | Exact Match |

---

## 2. Key Statistical & Structural Discoveries

### Discovery 1: Missingness is Completely Governed by the `Control` Multiplexer (MAR)
- **Zero Complete Rows:** There are **0 rows** with all 5 sensors present, and **0 rows** with all 5 sensors missing.
- **Row-Level Missingness:**
  - Exactly **3,437 rows (34.37%)** are missing 2 sensors.
  - Exactly **6,563 rows (65.63%)** are missing 3 sensors.
- **Deterministic Mechanism:** Sensor presence is a deterministic function of the `Control` configuration mode:
  - `Control == 'A'` (3,437 rows): Measures *Air Temp (100%)*, *Process Temp (100%)*, *Rotational Speed (100%)*. Missing: *Torque (0%)*, *Tool Wear (0%)*.
  - `Control == 'B'` (3,242 rows): Measures *Rotational Speed (100%)*, *Torque (100%)*. Missing: *Air Temp (0%)*, *Process Temp (0%)*, *Tool Wear (0%)*.
  - `Control == 'C'` (3,321 rows): Measures *Torque (100%)*, *Tool Wear (100%)*. Missing: *Air Temp (0%)*, *Process Temp (0%)*, *Rotational Speed (0%)*.
- **Perfect Co-Missingness:** `Air temperature (K)` and `Process temperature (K)` are **100% co-missing** ($r = 1.0$).
- **Statistical Significance:** Chi-Square test of `Control` vs Missingness yields $\chi^2 = 20,000.0, p < 10^{-300}$.
- **Mechanism Classification:** **Missing At Random (MAR)** conditional on `Control`.

### Discovery 2: Sensor Outliers Concentrate in Failure Classes (Critical Viva Point)
- **Torque Outlier Concentration:**
  - Across the 6,563 non-missing Torque records, the IQR method ($1.5 \times \text{IQR}$) identifies **64 outliers** ($< 12.0$ Nm or $> 67.2$ Nm).
  - **59 out of 64 outliers (92.2%) are actual failure events** (50 Power Failures, 9 Overstrain Failures, and only 5 normal operation records).
  - Deleting or aggressively trimming outliers would delete **60.2% of all Power Failure cases** in the entire dataset!
- **Rotational Speed Outlier Concentration:**
  - 286 IQR outliers ($> 2,094$ rpm), containing 31 Power Failure cases.
- **Symmetric Sensors:** `Air temperature`, `Process temperature`, and `Tool wear` have **0 IQR outliers** and **0 $Z > 3$ outliers**.
- **Takeaway:** Outliers represent catastrophic physical breakdown events, not measurement noise.

### Discovery 3: Extreme Class Imbalance ($508 : 1$)
- The majority class (`No failure`) represents **96.52%** of all data.
- The minority classes represent only **3.48% combined**:
  - `Heat Dissipation Failure`: 1.06% ($N = 106$)
  - `Overstrain Failure`: 0.98% ($N = 98$)
  - `Power Failure`: 0.83% ($N = 83$)
  - `Tool Wear Failure`: 0.42% ($N = 42$)
  - `Random Failures`: 0.19% ($N = 19$)
- **Imbalance Ratio:** $508 : 1$ (Majority to rarest class).
- **Modeling Requirement:** Accuracy is completely invalid (a dummy classifier achieves 96.52% accuracy with 0.0 macro-F1). Primary evaluation metrics must be **Macro-F1** and **Per-Class Recall**.

### Discovery 4: Physical Domain Failure Rules Hold with 100% Precision & Recall
Evaluating the domain failure criteria from the AI4I specification on the PMDI dataset confirms:
1. **Heat Dissipation Failure (HDF):** Condition $(Process\ Temp - Air\ Temp) < 8.6\text{ K} \land \omega < 1380\text{ rpm}$ is satisfied by exactly 106 rows, all 106 are HDF (100% precision, 100% recall). Monitored exclusively in `Control A`.
2. **Power Failure (PWF):** Condition $P = \tau \cdot \omega \cdot \frac{2\pi}{60} \notin [3500, 9000]\text{ W}$ is satisfied by exactly 83 rows, all 83 are PWF (100% precision, 100% recall). Monitored exclusively in `Control B`.
3. **Overstrain Failure (OSF):** Condition $Tool\ Wear \times \tau > \text{Threshold}(Type)$ is satisfied by exactly 98 rows, all 98 are OSF (100% precision, 100% recall). Monitored exclusively in `Control C`.
4. **Tool Wear Failure (TWF):** All 42 TWF cases fall strictly in the $198 - 246\text{ min}$ tool wear window in `Control C`.
5. **Unexplainable Failure Records:** **0 rows**. Every failure mode was monitored by the specific control configuration that captured its triggering sensors.

### Discovery 5: Strong Physical Sensor Correlations
- **Spindle Speed vs. Torque (Control B):** Pearson $r = -0.8588$, Spearman $\rho = -0.9210$. Inverse non-linear load curve reflecting mechanical power constraints ($P = \tau \cdot \omega$).
- **Air Temp vs. Process Temp (Control A):** Pearson $r = +0.8709$, Spearman $\rho = +0.8585$. Direct thermal equilibrium relationship.
- **Torque vs. Tool Wear (Control C):** Pearson $r = +0.0642$, Spearman $\rho = +0.0507$. Near-zero linear correlation, but their non-linear product determines tool structural failure.

### Discovery 6: Temporal Structure & Sampling Profile
- **Span:** February 27, 2014 to June 21, 2023 (~9.3 years) across 120 unique machines (`System` 0 to 119).
- **Sampling Gaps:** Mean interval 48.5 minutes, with irregular bursts and multi-month gaps.
- **UDI vs Date:** `UDI` is not monotonically sorted with `Date`.
- **Failure Distribution:** Failures are distributed uniformly across time, indicating stationary failure physics over the 9-year span.

---

## 3. Data Leakage Prevention Blueprint

| Risk Factor | Risk Level | Evidence from Dataset | Prevention Implementation |
| :--- | :--- | :--- | :--- |
| **Row Index Memorization** | High | `UDI` is an arbitrary sequential key ($1 \dots 10000$) | Drop `UDI` prior to feature matrix construction. |
| **Product Serial Memorization** | High | `Product ID` has format `[Type][Serial]` | Drop `Product ID`; preserve only `Type`. |
| **Imputer Leakage** | Critical | Imputing on full dataset leaks test distribution | Fit `IterativeImputer` / `KNNImputer` / `SimpleImputer` strictly on training folds. |
| **Class-Conditional Imputation** | Fatal | Imputing with knowledge of target $y$ | Imputation must be completely unsupervised. |
| **Resampling Leakage** | Fatal | Oversampling prior to train/test split | Integrate SMOTE inside `imblearn.pipeline.Pipeline` so resampling occurs only on training folds. |
| **Feature Selection Leakage** | High | Selection computed across train and test | Feature selectors fitted strictly inside cross-validation loops. |
| **Test Set Contamination** | Fatal | Evaluating on seen records | Stratified 80/20 train/test split locked before any transformation. |

---

## 4. Summary of Generated Figures in `reports/figures/`

1. `fig01_target_imbalance.png`: Diagnostic distribution in linear and logarithmic scale ($508:1$ imbalance).
2. `fig02_missingness_overview.png`: Column missingness percentages and row-level missing sensor counts ($2$ vs $3$).
3. `fig03_comissingness_by_control.png`: Sensor availability heatmap across Control modes (A, B, C).
4. `fig04_sensor_distributions_kde.png`: Continuous distributions, skewness, and KDE overlays for all 5 sensors.
5. `fig05_sensor_boxplots_by_diagnostic.png`: Stratified boxplots highlighting failure outlier concentrations.
6. `fig06_correlation_heatmaps.png`: Pairwise complete Pearson and Spearman correlation heatmaps.
7. `fig07_within_control_correlations.png`: Spindle Speed vs Torque (Control B) and Air Temp vs Process Temp (Control A).
8. `fig08_domain_physics_failure_boundaries.png`: AI4I failure rules vs PMDI ground truth points (HDF, PWF, OSF, TWF).
9. `fig09_categorical_failure_breakdown.png`: Failure distribution across `Type` (L, M, H) and `Control` (A, B, C).
10. `fig10_temporal_sampling_dynamics.png`: Annual failure counts and chronological sampling interval distributions.
