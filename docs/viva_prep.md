# Viva Voce Examination Defense Guide
**Course:** IT3051 - Fundamentals of Data Mining (SLIIT)  
**Project:** Machine Predictive Maintenance Diagnostic System  
**Group:** 05 - "Cognita"  
**Dataset:** AI4I-PMDI Predictive Maintenance Dataset  
**Target:** Multiclass Classification of Machine Health Diagnoses  

---

## 1. Quick-Fire Decision Justifications ("Why We Did This")

### 1. Why did we drop `UDI` and `Product ID`?
> **Viva Answer:** `UDI` is an arbitrary sequential integer ($1 \dots 10,000$), and `Product ID` is a unique machine serial whose prefix strictly duplicates the `Type` column. Keeping 1-to-1 identifiers provides zero generalizing power and causes models to memorize training rows, causing fatal data leakage.

### 2. Why did we use a Stratified Random Split instead of a Chronological Split?
> **Viva Answer:** Telemetry records span 9.3 years across 120 independent CNC machines with irregular multi-week sampling gaps, rather than a single continuous machine stream. A chronological split would concentrate early machines in train and later machines in test (causing covariate shift) and starve ultra-rare classes like Random Failures ($N=19$).

### 3. Why did we keep `Control` and one-hot encode it?
> **Viva Answer:** `Control` represents the physical diagnostic multiplexer mode (A, B, C) determining which sensor bus was actively recorded. It holds a statistically decisive relationship with the target ($\chi^2 = 659.08, p < 10^{-134}$) and provides critical context explaining why unmonitored sensors are absent.

### 4. Why did we NOT remove or trim extreme sensor outliers?
> **Viva Answer:** Outliers in predictive maintenance are genuine failure signatures rather than sensor noise. In our data, $92.2\%$ of Torque IQR outliers are actual breakdown events (50 Power Failures, 9 Overstrain Failures); removing outliers would delete $>60\%$ of all Power Failure instances in the dataset.

### 5. Why did we choose `Median + MissingIndicator` over KNN and Iterative Imputation?
> **Viva Answer:** In our Stratified 5-Fold CV benchmark, `Median + MissingIndicator` achieved the highest Macro-F1 score ($0.6785 \pm 0.0153$), required zero iterative convergence time, and executes in $O(1)$ constant time for real-time web application inference, whereas KNN requires heavy $O(N)$ distance computation against 8,000 training points.

### 6. Why did we engineer $\Delta T$, Mechanical Power, and Overstrain Product?
> **Viva Answer:** The underlying AI4I physical failure mechanics depend directly on thermal dissipation ($\Delta T < 8.6$ K), spindle power ($P < 3500$ W or $> 9000$ W), and tool structural strain ($\text{Wear} \times \text{Torque}$). Deriving these continuous interactions allows linear and tree models to capture failure boundaries directly, ranking as our top 3 most important features in permutation tests.

### 7. Why did we use Macro-F1 and Per-Class Recall instead of Accuracy?
> **Viva Answer:** The dataset suffers from severe $508 : 1$ class imbalance ($96.52\%$ normal operation vs $3.48\%$ failures). A naive baseline predicting "No failure" achieves $96.52\%$ accuracy with $0.0$ Macro-F1. Macro-F1 weights all 6 failure modes equally, ensuring rare catastrophic failures are reliably detected.

### 8. Why did we build a single Scikit-Learn Pipeline saved with `joblib`?
> **Viva Answer:** Packaging custom feature engineering, imputation, one-hot encoding, and scaling into ONE fitted `Pipeline` guarantees exact pre-processing parity between training and production web deployment, preventing pipeline drift and seamlessly transforming single-row inputs with NaNs.

---

## 2. 15 Comprehensive Viva Voce Questions & Empirical Answers

### Q1: What is the industrial problem, and what value does machine learning provide?
**Answer:** The project addresses industrial Predictive Maintenance (PdM) for milling machinery. Instead of relying on expensive scheduled maintenance or allowing catastrophic in-service tool breakage, our system performs multiclass diagnostic classification from streaming sensor telemetry. It classifies the machine condition into 6 states: *No failure*, *Heat Dissipation Failure*, *Overstrain Failure*, *Power Failure*, *Tool Wear Failure*, and *Random Failures*, allowing plant operators to execute targeted repairs before catastrophic downtime occurs.

### Q2: What are the key characteristics of the AI4I-PMDI dataset?
**Answer:** The AI4I-PMDI dataset (Autran, 2024; based on Matzka's AI4I 2020 dataset) contains 10,000 instances and 12 attributes. Unlike typical clean benchmarks, PMDI introduces realistic industrial sensor missingness: temperatures ($65.63\%$ missing), spindle speed ($33.21\%$ missing), torque ($34.37\%$ missing), and tool wear ($66.79\%$ missing). Non-sensor metadata has $0\%$ missing values.

### Q3: What is the target variable, and what makes this problem challenging?
**Answer:** The target variable is `Diagnostic`, containing 6 multiclass categories. The primary challenge is extreme class imbalance: `No failure` constitutes $96.52\%$ ($9,652$ instances), while all 5 failure modes comprise only $3.48\%$ ($348$ instances), with `Random Failures` having only $19$ instances ($0.19\%$, an imbalance ratio of $508:1$). This requires class-weighted objectives and macro-averaged metrics.

### Q4: What was the most surprising discovery during your Exploratory Data Analysis?
**Answer:** We discovered that sensor missingness is $100\%$ governed by the `Control` column ($A, B, C$). There are zero complete rows with all 5 sensors, and zero rows with all 5 missing. Every row has either exactly 2 missing sensors ($34.37\%$, Control A) or 3 missing sensors ($65.63\%$, Controls B & C). Mode A logs $[Air\ Temp, Process\ Temp, RPM]$, Mode B logs $[RPM, Torque]$, and Mode C logs $[Torque, Tool\ Wear]$. This proved the missingness is strictly **Missing At Random (MAR)** conditional on the sensor multiplexer channel.

### Q5: How did you prove the missing data mechanism is MAR rather than MCAR or MNAR?
**Answer:** In MCAR (Missing Completely at Random), missingness is independent of all observed and unobserved variables. We proved missingness is NOT MCAR because a Chi-Square test of `Control` against missingness yields $\chi^2 = 20,000.0, p < 10^{-300}$ (deterministic relationship). Furthermore, it is NOT MNAR (Missing Not at Random) because the probability of missingness does not depend on the unobserved value itself (e.g. high temperatures are not suppressed), but on the known logging multiplexer channel `Control`.

### Q6: Why is standard outlier trimming fatal in predictive maintenance?
**Answer:** In anomaly detection and predictive maintenance, outliers are physical indicators of failure rather than measurement corruption. In our dataset, the IQR method identifies 64 outliers in `Torque (Nm)`. Empirical analysis reveals that **$59$ out of the $64$ outliers ($92.2\%$) are actual failure cases** ($50$ Power Failures, $9$ Overstrain Failures, only $5$ normal operations). Removing outliers would delete over $60\%$ of all Power Failure instances, blinding the model to critical breakdowns.

### Q7: Why did you choose an 80/20 Stratified Random Split over a Chronological Split?
**Answer:** A chronological split is only appropriate for single-stream autoregressive time-series forecasting. In PMDI, the records represent asynchronous, irregular telemetry snapshots across 120 separate machines over 9.3 years with multi-week gaps. A chronological split would place early machines in train and later machines in test, introducing severe machine-level covariate shift and failing to guarantee enough samples for ultra-rare classes like Random Failures ($N=19$). A stratified random split preserves exact class proportions ($96.53\%$ train vs $96.50\%$ test) while ensuring zero train-test leakage.

### Q8: What empirical results justified selecting Median Imputation over KNN or Iterative Imputer?
**Answer:** We conducted a Stratified 5-Fold Cross-Validation benchmark on `X_train` with a fixed baseline RandomForest model. The empirical results:
- **Median + MissingIndicator:** Macro-F1 = **$0.6785 \pm 0.0153$**
- **IterativeImputer + MissingIndicator:** Macro-F1 = **$0.6767 \pm 0.0175$** (with convergence warnings)
- **Median (Raw):** Macro-F1 = **$0.6747 \pm 0.0111$**
- **KNNImputer (k=5) + MissingIndicator:** Macro-F1 = **$0.6690 \pm 0.0171$**
- **KNNImputer (k=5):** Macro-F1 = **$0.6687 \pm 0.0198$**
`Median + MissingIndicator` achieved the highest score, zero convergence failure risk, and instantaneous $O(1)$ computation for web app inference.

### Q9: What domain physics features did you engineer, and why?
**Answer:** We developed `PhysicsFeatureEngineer` to compute 4 continuous features based on milling mechanics:
1. **$\Delta T = Process\ Temp - Air\ Temp$:** Captures heat dissipation capacity (HDF rule: $\Delta T < 8.6$ K).
2. **$Mechanical\ Power\ (W) = Torque \times \omega \times \frac{2\pi}{60}$:** Captures spindle power overload/underload (PWF rule: $P \notin [3500, 9000]$ W).
3. **$Overstrain\ Product = Tool\ Wear \times Torque$:** Captures structural mechanical stress (OSF rule: $\text{Product} > 11000 - 13000$ min$\cdot$Nm).
4. **$Missing\ Sensors\ Count$:** Quantifies operational sensor coverage.
We derived continuous interactions rather than hardcoding binary rules so the model retains smooth probability boundaries.

### Q10: What were your feature importance and feature selection findings?
**Answer:** Using Permutation Importance on the training fold, our engineered physics features dominated the rankings:
1. `Mechanical_Power_W` (Importance: $0.1920$)
2. `Overstrain_Product` (Importance: $0.1721$)
3. `Rotational speed (rpm)` (Importance: $0.1616$)
4. `Tool wear (min)` (Importance: $0.1548$)
5. `Temp_Difference` (Importance: $0.1493$)
The engineered physical interactions proved to be the single most decisive predictors of machine failure across all evaluation metrics.

### Q11: How did you prevent Data Leakage throughout Stage 3 and Stage 4?
**Answer:** We implemented a rigorous 6-point leakage prevention protocol:
1. **Split First:** The 80/20 train/test split was locked before fitting any imputer, encoder, or scaler.
2. **Identifier Elimination:** Dropped `UDI` and `Product ID` to prevent index memorization.
3. **Unsupervised Imputation:** Imputation was strictly unsupervised; target labels $y$ were never used.
4. **Resampling Isolation:** SMOTE was embedded inside cross-validation pipelines so oversampling only touched training folds.
5. **Feature Selection Isolation:** Permutation importance and MI were computed exclusively on `X_train`.
6. **Isolated Test Set:** `X_test` remained untouched and unseen until final evaluation.

### Q12: How did you prepare for Class Imbalance handling?
**Answer:** We benchmarked two leak-free approaches inside our cross-validation pipeline on `X_train`:
- **Class Weighting (`class_weight='balanced'`):** Penalizes misclassifications inversely proportional to class frequencies, achieving **$0.6785$ Macro-F1**.
- **SMOTE ($k=3$):** Synthesizes minority failure instances inside training folds, achieving **$0.6874$ Macro-F1**.
Both strategies effectively overcome majority class bias without distorting the test distribution.

### Q13: Why is feature scaling applied conditionally inside the pipeline?
**Answer:** Tree-based models (RandomForest, LightGBM, XGBoost) are invariant to monotonic feature scale and benefit from unscaled physical units for interpretability. In contrast, distance-based and linear models (SVM, KNN, Logistic Regression) are severely distorted by differing units (e.g. Watts in thousands vs Torque in tens). Our `build_preprocessor()` supports conditional `StandardScaler` integration tailored to the downstream classifier.

### Q14: How does the fitted preprocessor handle single-row web app inputs with missing values?
**Answer:** When a web app user enters partial sensor telemetry (e.g. only Temperature and Speed, leaving Torque and Tool Wear empty as NaN), our saved `preprocessor.joblib` pipeline:
1. Calculates domain interaction features ($\Delta T$, Power, Overstrain).
2. Generates missing indicator flags ($0$ for observed, $1$ for missing).
3. Imputes missing fields with the median learned during training.
4. One-hot encodes categorical machine attributes (handling unseen types gracefully).
5. Outputs a clean 23-dimensional feature vector with zero NaNs ready for immediate model prediction.

### Q15: Why is per-class recall critical alongside Macro-F1 for industrial maintenance?
**Answer:** In industrial predictive maintenance, the cost of a False Negative (failing to detect an imminent catastrophic failure) is orders of magnitude higher than a False Positive (conducting an unnecessary inspection). Tracking per-class recall guarantees that ultra-rare failure modes (such as Tool Wear Failure at $N=42$ or Random Failures at $N=19$) achieve high sensitivity and are not masked by aggregate metrics.

---

## 3. Open Decisions for Group Discussion

1. **Classifier Selection in Stage 5:**
   - Benchmark tree ensembles (RandomForest, LightGBM, CatBoost) vs Cost-Sensitive Multi-Layer Perceptrons / Support Vector Machines.
2. **Final Imbalance Strategy:**
   - Decide between algorithmic cost-weighting (`class_weight='balanced'`) vs synthetic sampling (`SMOTE` with $k=3$) based on validation per-class recall.
3. **Threshold Tuning for Web Deployment:**
   - Optimize class probability decision thresholds to maximize recall on high-consequence failure modes (HDF, PWF, OSF).
