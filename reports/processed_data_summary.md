# Processed Dataset Inspection & Integrity Summary
**Course:** IT3051 - Fundamentals of Data Mining (SLIIT)  
**Project:** Predictive Maintenance Diagnostic System  
**Dataset:** AI4I-PMDI Maintenance Dataset  
**Stage:** Stage 4 (Data Preprocessing & Feature Engineering)  

---

## 1. Dataset Dimensions & Storage Location

The processed datasets have been exported and verified in the local system storage:
- **Base Directory:** `C:\Users\oshani\AppData\Local\Temp\`

| File Name | Row Count | Column Count | Description |
| :--- | :---: | :---: | :--- |
| `X_train_processed.csv` | 8,000 | 23 | Final model-ready training feature matrix |
| `X_test_processed.csv` | 2,000 | 23 | Final model-ready test feature matrix |
| `y_train.csv` | 8,000 | 1 | Target label for training (`Diagnostic`) |
| `y_test.csv` | 2,000 | 1 | Target label for testing (`Diagnostic`) |
| `train_engineered.csv` | 8,000 | 20 | Interpretable version (imputed + physics, pre-OHE) |
| `test_engineered.csv` | 2,000 | 20 | Interpretable version (imputed + physics, pre-OHE) |

---

## 2. Final 23-Dimensional Feature Representation

The final model-ready matrix consists of exactly 23 features decomposed across 4 functional origins:

### A. Imputed Numerical Features (9 features)
1. `Air temperature (K)`
2. `Process temperature (K)`
3. `Rotational speed (rpm)`
4. `Torque (Nm)`
5. `Tool wear (min)`
6. `Temp_Difference` ($\Delta T = T_{proc} - T_{air}$)
7. `Mechanical_Power_W` ($P = \tau \cdot \omega \cdot \frac{2\pi}{60}$)
8. `Overstrain_Product` ($\text{Wear} \times \tau$)
9. `Missing_Sensors_Count`

### B. Binary Missing Indicators (8 features)
10. `missing_ind_Air temperature (K)`
11. `missing_ind_Process temperature (K)`
12. `missing_ind_Rotational speed (rpm)`
13. `missing_ind_Torque (Nm)`
14. `missing_ind_Tool wear (min)`
15. `missing_ind_Temp_Difference`
16. `missing_ind_Mechanical_Power_W`
17. `missing_ind_Overstrain_Product`

### C. One-Hot Encoded Categoricals (6 features)
18. `Type_H`
19. `Type_L`
20. `Type_M`
21. `Control_A`
22. `Control_B`
23. `Control_C`

---

## 3. Missing Value Audit

| Matrix | Raw Sensor Missing Rate | Processed Missing Count | Status |
| :--- | :---: | :---: | :---: |
| `X_train_processed` | 33.2% - 66.8% | **0** | Clean (100% complete) |
| `X_test_processed` | 33.2% - 66.8% | **0** | Clean (100% complete) |
| `train_engineered` | 33.2% - 66.8% | **0** | Clean (100% complete) |
| `test_engineered` | 33.2% - 66.8% | **0** | Clean (100% complete) |

---

## 4. Target Class Distribution (`Diagnostic`)

| Class | Full Dataset (N=10,000) | Train Split (N=8,000) | Test Split (N=2,000) |
| :--- | :---: | :---: | :---: |
| `No failure` | 9,652 (96.52%) | 7,722 (96.53%) | 1,930 (96.50%) |
| `Heat Dissipation Failure` | 106 (1.06%) | 85 (1.06%) | 21 (1.05%) |
| `Overstrain Failure` | 98 (0.98%) | 78 (0.97%) | 20 (1.00%) |
| `Power Failure` | 83 (0.83%) | 66 (0.83%) | 17 (0.85%) |
| `Tool Wear Failure` | 42 (0.42%) | 34 (0.43%) | 8 (0.40%) |
| `Random Failures` | 19 (0.19%) | 15 (0.19%) | 4 (0.20%) |

---

## 5. Data Integrity & Leakage Prevention Assertions

1. **Strict 80/20 Stratification:** Verified. Both partitions match target class proportions within $<0.05\%$ margin of error.
2. **Leak-Free Transformation:** Verified. All imputation medians, indicators, and encoders were strictly fitted on `X_train` only; `X_test` was only passed through `.transform()`.
3. **No Row Index Columns:** Verified. Output was exported without artificial index columns (`index=False`).
4. **No Identifiers/Target in X:** Verified. `UDI`, `Product ID`, `Date`, and `System` dropped. `Diagnostic` isolated in `y`.
