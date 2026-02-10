# Telco Customer Churn Predictor

---

## 1. Project Overview
- This project studies customer churn in a telecom environment using historical service and subscription data.
- The objective is to predict whether a customer is likely to discontinue service and to understand which behavioural factors are associated with higher exit risk.
- Two models are evaluated: a transparent baseline using Logistic Regression and a more complex ensemble approach using Random Forest.
- The comparison helps assess whether increased model complexity leads to meaningful improvements in identifying at-risk customers.
- The insights derived from this analysis can support targeted retention strategies, proactive intervention, and informed decision-making.

---

## 2. Objectives
- Predict whether a customer will churn (Yes/No).
- Establish a baseline model.
- Evaluate whether increased model complexity improves minority detection.
- Analyze practical implications of prediction errors.

---

## 3. Dataset
- Source: [Kaggle Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data)
- Target variable: `Churn`

---

## 4. Data Preprocessing
The following steps were applied:

- Removed customer identifiers.
- Converted `TotalCharges` to numeric format.
- Removed customers with zero tenure due to insufficient service history.
- Applied one-hot encoding for categorical variables.
- Checked and removed duplicate entries.

---

## 5. Exploratory Data Analysis (EDA)

### A. Contract Duration vs Churn
<img width="700" height="593" alt="image" src="https://github.com/user-attachments/assets/741e6669-138d-4314-9953-9be27b931098" />

**Insight:**  
Customers with long-term contracts demonstrate significantly lower churn probability, indicating commitment duration is a major retention factor.

---

### B. Automatic Credit Card Payments vs Churn
<img width="713" height="588" alt="image" src="https://github.com/user-attachments/assets/03061a80-7d17-4b3e-9c9a-58f727da953c" />

**Insight:**  
Automatic payment enrollment is associated with reduced churn, possibly due to reduced friction and stronger service continuity.

---

### C. Multiple Lines vs Churn
<img width="702" height="591" alt="image" src="https://github.com/user-attachments/assets/500d87bc-e56b-4381-9655-82af1fcfbb75" />

**Insight:**  
The association is present but modest, suggesting this variable may contribute to risk but is not a dominant driver.

---

## 6. Train–Validation Split
- 80% training, 20% validation.
- Stratified sampling to preserve churn ratio.
- Random state = 42.

---

## 7. Models Evaluated

### Logistic Regression (Baseline)
Chosen for transparency, interpretability, and auditability.

### Random Forest (Ensemble Alternative)
Evaluated to determine whether non-linear modeling improves detection.

---

## 8. Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1 Score  

Given class imbalance, **recall and F1** were considered especially important.

---

## 9. Results
<img width="1040" height="678" alt="image" src="https://github.com/user-attachments/assets/673a2501-ba1a-4273-849a-828605606b61" />

---

## 10. Performance Comparison
<img width="1031" height="674" alt="image" src="https://github.com/user-attachments/assets/c11cd6bc-9cdb-40ea-8f61-531ed3524a05" />

---

## 11. Confusion Matrices

### Logistic Regression
<img width="579" height="458" alt="image" src="https://github.com/user-attachments/assets/8687a3d5-52ef-4f64-bad5-3213a20c6563" />

### Random Forest
<img width="563" height="456" alt="image" src="https://github.com/user-attachments/assets/6ac37511-b334-4575-b149-cd10c1028bb1" />

---

## 12. Key Observations
- Logistic Regression achieved comparable or better recall.
- Increased complexity did not automatically yield improved minority detection.
- Simpler, interpretable models may remain strong candidates in regulated or governance-sensitive environments.
