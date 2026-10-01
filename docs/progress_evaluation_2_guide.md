# Progress Evaluation 2 (30%) Viva Voce Examination Defense Guide
**Course:** IT3051 - Fundamentals of Data Mining (SLIIT)  
**Project:** Machine Learning-Based Predictive Maintenance System for Industrial Equipment  
**Group:** 05 - "Cognita"  
**Dataset:** AI4I-PMDI Predictive Maintenance Dataset  
**Focus:** Individual Technical Defense for Stages 6, 7 & 8 (Modelling, Optimization & Selection)  

---

## 1. Individual Technical Ownership & Key Contributions

When presenting your work individually during Progress Evaluation 2, speak from first-person authority using specific technical terminology:

### A. Architectural & Experimental Design
1. **Designed the Multi-Model Comparison Framework:** Implemented 5 diverse algorithm families (Multinomial Logistic Regression, Cost-Sensitive RBF SVC, Random Forest, Extra Trees, and HistGradientBoosting) to contrast linear boundaries, kernel maximum margins, bagging ensembles, and histogram-based boosting.
2. **Engineered the Leakage-Free Validation Engine:** Constructed a strict `StratifiedKFold` (5-fold) validation architecture ensuring that scaling, imputation, and oversampling were computed solely within training folds.
3. **Formulated the Controlled Feature Ablation Experiment:** Designed Experiment A (raw sensors) vs. Experiment B (domain physics interactions) to empirically prove that engineering $\Delta T$, Mechanical Power $W$, and Overstrain Product produces a **$+14.89\%$ lift in Macro-F1**.
4. **Resolved Severe Class Imbalance:** Benchmarked algorithmic cost-weighting (`class_weight='balanced_subsample'`) against SMOTE inside cross-validation, demonstrating that cost-weighting yields higher recall ($78.3\%$ vs $68.9\%$) with zero synthetic artifact noise.
5. **Systematic Hyperparameter Tuning:** Formulated the parameter search spaces and executed `RandomizedSearchCV` on Macro-F1, lifting the champion Random Forest from $0.6744$ to **$0.7000$ Macro-F1**.
6. **Built the Production Deployment Pipeline:** Integrated the fitted preprocessor with the champion estimator into a single serialized `champion_pipeline.joblib`, enabling single-row web inference with missing sensors in $0.4$ ms.

---

## 2. Deep Technical Breakdown by Stage

### A. Algorithm Selection & Justification

| Algorithm | Mathematical Paradigm | Inductive Bias | Strengths in PMDI | Weaknesses / Trade-offs | Role in Project |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Multinomial Logistic Regression** | Parametric Linear Probabilistic Model | Linear decision boundaries in log-odds space | Interpretable coefficients; rapid convergence | Cannot capture non-linear spindle power boundaries without interaction terms | Linear parametric baseline |
| **Support Vector Classifier (SVC)** | Maximum-Margin Kernel Method (RBF) | Non-linear geometric separation in dual Hilbert space | High margin separation; effective on continuous sensor manifolds | High $O(N^2)$ fit time; struggles with extreme class imbalance without calibrated probability thresholds | Non-linear kernel baseline |
| **Random Forest Classifier** | Bagging Ensemble of Deep Trees | Orthogonal recursive binary partitioning | Captures discontinuous failure thresholds; scale invariant; robust to outliers; balanced subsampling handles rare classes | Large model size ($~2$ MB); moderate memory consumption | **Champion Model** |
| **Extra Trees Classifier** | Extremely Randomized Bagging | Purely randomized split threshold selection | Lower variance than standard RF; extremely fast training | Random splits often miss narrow geometric failure pockets of rare classes | Variance-reduction baseline |
| **HistGradientBoosting** | Sequential Gradient Boosted Decision Trees | Additive gradient boosting in function space using histogram binning | Fast training; natively handles missing values; high precision on structured tabular data | Higher cross-fold variance ($\sigma=0.031$); susceptible to label noise in ultra-rare classes | Boosting contender |

---

### B. Validation Strategy

#### Why Stratified 5-Fold Cross-Validation?
1. **Mathematical Necessity:** In our training split ($8,000$ instances), the rarest class (`Random Failures`) has only $15$ samples. A 5-fold split places **exactly 3 instances in each validation fold**. If 10 folds were used, folds would contain only 1 or 2 instances, causing fold-level recall to jump wildly between $0\%$, $50\%$, and $100\%$.
2. **Untouched Test Set Isolation:** The 2,000-instance test set was locked before any feature engineering, model training, or hyperparameter search was conducted. It was evaluated strictly ONCE after all engineering decisions were frozen.
3. **Leakage Elimination:** In every fold iteration, scaling, imputation, and oversampling were strictly fitted on the 4 training folds and applied to the 5th validation fold via `sklearn.pipeline.Pipeline`.

---

### C. Evaluation Metrics Rationale

#### Why Ordinary Accuracy is Scientifically Invalid
In the AI4I-PMDI dataset, normal operations constitute $96.52\%$ of all records. A naive trivial classifier predicting "No failure" for every machine achieves **$96.52\%$ accuracy**, but has an F1-score of **$0.00$ on all failure modes**, completely failing to detect any breakdown.

#### Golden Standards for Industrial Maintenance:
1. **Macro F1-Score (Primary Metric):**
   $$\text{Macro F1} = \frac{1}{6} \sum_{k=1}^6 \text{F1}_k$$
   Weights all 6 classes equally. Detecting a rare tool failure ($N=34$) is valued just as much as classifying the majority normal class ($N=7,722$).
2. **Balanced Accuracy & Macro Recall:**
   $$\text{Balanced Accuracy} = \frac{1}{6} \sum_{k=1}^6 \text{Recall}_k$$
   Reflects the average sensitivity of the monitoring system across failure types.
3. **Per-Class Recall:** Tracks sensitivity for each individual failure mode (e.g. $100\%$ on PWF and HDF, $95\%$ on OSF).

---

### D. Hyperparameter Tuning & Optimization Decisions

#### Search Strategy: `RandomizedSearchCV` on Macro-F1
- **Why RandomizedSearchCV over Exhaustive GridSearch?** With 6 parameters across 5 folds ($75$ fits per model), Randomized Search explores continuous and discrete parameter spaces efficiently without wasting computation on unpromising hyperparameter grids.
- **Random Forest Key Parameters Tuned:**
  - `n_estimators`: Increased from $150 \rightarrow 200$ to stabilize tree voting variance.
  - `max_depth`: Constrained to $10$ (prevented individual trees from overfitting to rare noise instances).
  - `min_samples_split`: Set to $6$ and `min_samples_leaf` to $2$ to regularize leaf purity.
  - `max_features`: Set to $0.8$ (sub-sampling 18 out of 23 features per split, ensuring decorrelation while capturing multi-sensor interactions).
  - `class_weight`: Set to `'balanced_subsample'` (re-computes class weights for each bootstrap sample, providing dynamic balance across individual trees).

---

### E. Optimization Decisions & Empirical Findings

1. **Feature Engineering Impact:**
   - Experiment A (Raw) Macro F1: $0.5870$
   - Experiment B (Raw + Physics) Macro F1: **$0.6744$** ($+14.89\%$ relative lift).
   - *Technical Explanation:* Deriving continuous interactions ($T_{proc} - T_{air}$, $\tau \cdot \omega$, and $Wear \times \tau$) transforms complex curvilinear hyper-surfaces into orthogonal linear splits that decision trees can isolate in a single cut.
2. **Imbalance Handling Decisions:**
   - Unweighted Random Forest: $0.6482$ Macro F1, $0.6498$ Recall.
   - Class Weighting (`'balanced'`): **$0.6744$ Macro F1, $0.7828$ Recall** ($\sigma=0.0124$).
   - SMOTE ($k=2$): $0.6869$ Macro F1, $0.6899$ Recall ($\sigma=0.0352$).
   - *Technical Decision:* Class weighting was chosen because it achieved a **$+9.3\%$ higher Macro Recall** and **$3\times$ lower cross-fold variance** than SMOTE without creating synthetic noise in borderline feature spaces.
3. **Feature Selection Decision:**
   - Permutation importance revealed that the top 5 features are: `Tool wear (min)`, `Mechanical_Power_W`, `Overstrain_Product`, `Temp_Difference`, and `Rotational speed (rpm)`.
   - While Top 10 features achieved $0.6815$ Macro F1, we retained the full 23-feature set because the binary missing indicators provide essential operational context for web application inputs when sensor telemetry is partially unmonitored.

---

### F. Final Model Selection & Unseen Test Performance

- **Champion Model:** Tuned Random Forest Classifier (`n_estimators=200`, `max_depth=10`, `class_weight='balanced_subsample'`).
- **Performance on Untouched Test Partition ($N=2,000$):**
  - **Accuracy:** $98.50\%$
  - **Macro F1-Score:** **$0.7101$**
  - **Balanced Accuracy / Macro Recall:** **$0.7606$**
  - **Heat Dissipation Failure (HDF):** Precision $1.00$, Recall **$1.00$**, F1 **$1.00$** ($21/21$ detected)
  - **Power Failure (PWF):** Precision $1.00$, Recall **$1.00$**, F1 **$1.00$** ($17/17$ detected)
  - **Overstrain Failure (OSF):** Precision $1.00$, Recall **$0.95$**, F1 **$0.97$** ($19/20$ detected)
  - **Tool Wear Failure (TWF):** Precision $0.19$, Recall **$0.625$**, F1 $0.29$ ($5/8$ detected)
  - **Random Failures (RNF):** Precision $0.00$, Recall $0.00$, F1 $0.00$ ($0/4$ detected)

---

## 3. Anticipated Viva Voce Examiner Questions & High-Scoring Answers

### Q1: Why did you test five different algorithms rather than immediately choosing Random Forest or XGBoost?
**Model Answer:**  
"A core principle of empirical machine learning is the 'No Free Lunch' theorem. We selected five models representing distinct inductive biases: Logistic Regression establishes a linear parametric baseline; Support Vector Classifiers test maximum-margin separation in dual space; Random Forest and Extra Trees test de-correlated and randomized bagging; and HistGradientBoosting evaluates sequential gradient boosting. Comparing these paradigms allowed us to understand the underlying geometry of the problem: predictive maintenance failure modes are defined by non-linear physical thresholds rather than smooth hyperplanes, which is why tree-based bagging decisively outperformed linear and kernel methods."

---

### Q2: Why did you prioritize Macro-F1 and Balanced Accuracy over standard Accuracy?
**Model Answer:**  
"Our dataset exhibits severe class imbalance: $96.52\%$ of instances are normal operations, while failure modes comprise only $3.48\%$ combined, with Random Failures having only $15$ instances in training ($514:1$ ratio). Under standard accuracy, a trivial dummy model that predicts 'No failure' for every sample scores $96.52\%$, but fails to detect a single breakdown. Macro-F1 treats all six classes with equal weight, penalizing models that ignore minority failures. In industrial maintenance, undetected failures cause catastrophic plant downtime, so macro-averaged sensitivity is our primary performance criterion."

---

### Q3: How did you ensure your validation strategy was completely free of data leakage?
**Model Answer:**  
"We enforced a strict three-tier leakage prevention protocol:
First, we locked an 80/20 Stratified Random Split before fitting any preprocessing component.
Second, the 2,000-instance test set was kept completely untouched and was only evaluated once after all hyperparameters, features, and model selections were finalized.
Third, within our Stratified 5-Fold cross-validation, all feature engineering, imputation, and scaling were executed inside an `sklearn.pipeline.Pipeline` fitted strictly on the four training folds of each iteration. Information from validation folds never contaminated training."

---

### Q4: Why did your feature engineering ablation experiment show such a dramatic lift?
**Model Answer:**  
"In our controlled ablation experiment, adding domain physics features lifted the Macro-F1 of Random Forest from $0.5870$ to $0.6744$—a $+14.89\%$ relative increase. This occurred because milling failure modes in AI4I are governed by thermodynamic and kinematic laws:
Power Failure depends on spindle power $P = \tau \cdot \omega \cdot \frac{2\pi}{60}$ violating $[3500, 9000]$ W;
Heat Dissipation Failure depends on $\Delta T = T_{proc} - T_{air} < 8.6$ K;
and Overstrain depends on the product of Tool Wear and Torque.
By explicitly engineering these continuous interaction terms, the decision trees did not have to approximate non-linear hyperbolic surfaces through hundreds of stair-step cuts; the physical failure boundary became a single orthogonal split."

---

### Q5: Why did you choose class weighting over SMOTE for your final model?
**Model Answer:**  
"While SMOTE achieved a comparable Macro-F1 ($0.6869$ vs $0.6744$), our empirical fold analysis revealed two critical drawbacks:
First, SMOTE produced significantly higher fold-to-fold variance ($\sigma=0.0352$ vs $\sigma=0.0124$). For our rarest class, Random Failures, synthesizing artificial instances between only 12 training samples in high-dimensional space created noisy points in normal operating regions.
Second, algorithmic class weighting (`class_weight='balanced_subsample'`) achieved a substantially higher Macro Recall ($0.7828$ vs $0.6899$). By penalizing misclassifications inversely proportional to class frequencies directly in the tree loss function, we boosted minority failure detection without synthesizing artificial artifacts."

---

### Q6: Why did Random Failures achieve 0.00 F1 on the test set across all models?
**Model Answer:**  
"This is a fundamental domain characteristic of the AI4I dataset rather than a modelling flaw. By definition, 'Random Failures' represent stochastic anomalies—such as an unexpected power surge or external physical impact—that occur independently of operating telemetry. The sensor values for Random Failures lie completely within the normal operating distributions of speed, torque, and temperature. Because there is no distinguishing telemetry signature, any attempt to force the model to predict Random Failures would cause severe false positive alarms across the 9,652 normal operating machines."

---

### Q7: Why did Tuned Random Forest outperform HistGradientBoosting as the Champion Model?
**Model Answer:**  
"Although baseline HistGradientBoosting had a slightly higher initial Macro-F1 ($0.6830$ vs $0.6744$), hyperparameter tuning via `RandomizedSearchCV` elevated Random Forest to **$0.7000$ Macro-F1**, outperforming tuned HistGradientBoosting ($0.6937$).
More importantly, Random Forest demonstrated **over $50\%$ lower cross-fold variance** ($\sigma=0.0101$ vs $\sigma=0.0224$) and higher sensitivity on critical breakdowns ($98.3\%$ vs $96.8\%$). In an industrial predictive maintenance system, stability and sensitivity on rare breakdowns are vastly preferred over aggressive gradient boosting that exhibits greater fold volatility."

---

### Q8: How is the final model prepared for real-time web application inference?
**Model Answer:**  
"We packaged the fitted Stage 4 `preprocessor` and the tuned Champion Random Forest into a unified Scikit-Learn `Pipeline`, serialized to `models/champion_pipeline.joblib`. When a user submits partial telemetry from a web browser (e.g. only rotational speed and torque, leaving temperatures empty as NaNs), the pipeline automatically:
1. Calculates domain physics features ($\Delta T$, Spindle Power, Overstrain Product).
2. Generates binary missing indicators.
3. Imputes missing fields with training medians.
4. One-hot encodes operational modes (`Type` and `Control`).
5. Generates the multiclass diagnosis and class probabilities in $0.4$ ms.
We verified this in our test suite, where a single-row input with NaNs was instantly diagnosed as 'Power Failure' with $100.00\%$ probability."

---

## 4. Project Limitations & Future Improvements

### Current Limitations:
1. **Stochastic Anomaly Inseparability:** The current sensor suite cannot anticipate 'Random Failures' due to the absence of high-frequency acoustic or vibration telemetry.
2. **Snapshot Telemetry Assumption:** The system treats each telemetry record as an independent snapshot rather than modelling temporal degradation trajectories over time.
3. **Tool Wear Failure Recall:** Tool Wear Failure achieved $62.5\%$ recall on the test set due to borderline overlap with standard worn tooling.

### Future Improvements:
1. **Vibration Spectrum Telemetry:** Integrate fast Fourier transform (FFT) vibration sensor streams to detect micro-cracks before thermal or torque signatures appear.
2. **Cost-Sensitive Threshold Tuning:** Adjust class decision probability thresholds using an explicit dollar-loss cost matrix (e.g. assigning a $\$10,000$ penalty to False Negatives).
3. **Sequential Degradation Modelling:** Implement temporal models (e.g. LSTMs or Temporal Convolutional Networks) to predict Remaining Useful Life (RUL) alongside snapshot multiclass diagnosis.
