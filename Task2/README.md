# Credit Card Fraud Detection Pipeline

An end-to-end Machine Learning pipeline developed in Python to detect fraudulent credit card transactions. This project focuses on handling extreme class imbalance (~0.17% fraud rate) using advanced sampling techniques, pipeline scaling, and careful hyperparameter optimization under strict computational constraints.

## 📌 Project Overview
Fraud detection is a classic "needle in a haystack" problem. In this project, we built and evaluated two foundational classification models:
1. **Logistic Regression** (Optimized via `GridSearchCV`)
2. **Random Forest Classifier** (Optimized via `RandomizedSearchCV`)

Given the operational limits of training complex models on standard consumer hardware, the search space was strategically downscaled (3-fold cross-validation, 8 random combinations for Random Forest) to demonstrate that robust predictive power can be achieved efficiently.

---

## 📊 Dataset & Class Imbalance
The dataset contains transactions made by credit cards, where we encounter:
* **Legitimate Transactions:** 56,864 (99.83%)
* **Fraudulent Transactions:** 98 (0.17%)

To prevent data leakage, the dataset was strictly split into training and test sets *before* applying any synthetic oversampling (`SMOTE`) or scaling techniques.

---

## 🛠️ Pipeline & Modeling Architecture
The project utilizes `imblearn.pipeline.Pipeline` to ensure all preprocessing and resampling steps are encapsulated perfectly within each cross-validation fold:

1. **Robust Scaling:** RobustScaler was implemented to handle features with heavy outliers safely.
2. **Resampling:** Combined SMOTE techniques were utilized to intelligently handle the minority class without overfitting.
3. **Hyperparameter Tuning:**
   * **Logistic Regression:** Grid search optimized regularizations ($L_1$ vs $L_2$) and the inverse regularization strength `C`.
   * **Random Forest:** Randomized search tuned tree depths, split criteria, and estimator counts over bounded iterations to save CPU clock cycles.

---

## 📈 Model Performance & Results

Since standard accuracy is highly misleading for imbalanced datasets, models were evaluated using **Recall**, **Precision**, and the **Area Under the Precision-Recall Curve (AUPRC)**.

### Performance Summary

| Evaluation Metric | Logistic Regression | Random Forest | Winner |
| :--- | :---: | :---: | :---: |
| **Recall (Catch Rate)** | **0.9184** | 0.8571 | Logistic Regression |
| **Precision (Exactness)** | 0.0594 | **0.4221** | Random Forest |
| **F1-Score (Balance)** | 0.1117 | **0.5657** | Random Forest |
| **ROC-AUC Score** | 0.9634 | **0.9782** | Random Forest |
| **AUPRC (PR Curve Area)**| 0.7212 | **0.7998** | Random Forest |

### 🧠 Model Selection Justification
The **Random Forest Classifier** was selected as the final production champion:
* **Operational Efficiency:** While Logistic Regression caught 92% of frauds, its abysmal precision (6%) means it flags roughly **15 false alarms for every 1 true fraud instance**, creating an unmanageable alert volume for a business. 
* **The Precision-Recall Advantage:** Random Forest keeps a strong catch rate (86%) while drastically pushing precision up to 42%, yielding a commanding **AUPRC of 0.7998** ($+0.078$ over Logistic Regression).

---

## 💾 Saving & Reusing the Model
The final optimized Random Forest pipeline has been exported and compressed using `joblib` for rapid deployment and future inference.

### How to Load and Use the Champion Model:
```python
import joblib

# Load the deployment asset
loaded_fraud_model = joblib.load('final_fraud_detection_rf_model.joblib')

# Feed raw, unscaled features matching the training shape
# predictions = loaded_fraud_model.predict(X_new)
# probabilities = loaded_fraud_model.predict_proba(X_new)[:, 1]