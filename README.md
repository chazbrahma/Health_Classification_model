# Healthcare ML Classification Project

This project applies machine learning to classify patients into three diagnostic categories — Abnormal, Inconclusive, and Normal — based on anonymized healthcare features.

---

## Problem Statement

Our goal is to predict test result outcomes using structured patient data.

### Target Classes:
- Abnormal
- Inconclusive
- Normal

---

## Workflow Overview

### 1. Data Cleaning
- Dropped irrelevant columns (Name, Room Number, etc.)
- Created `Length_of_Stay` and `Age_Billing` (interaction term)

### 2. Exploratory Data Analysis (EDA)

**Test Result Balance:**  
![Test Results](https://github.com/chazbrahma/Health_Classification_model/blob/main/Plots/test_results_distribution%20(1).png?raw=true)

**Age Histogram:**  
![Age Histogram](https://github.com/chazbrahma/Health_Classification_model/blob/main/Plots/age_distribution.png?raw=true)

**Medical Condition Distribution:**  
![Medical Conditions](https://github.com/chazbrahma/Health_Classification_model/blob/main/Plots/medical_condition_distribution.png?raw=true)

---

## Feature Engineering

- One-hot encoding of categorical variables
- Derived interaction: `Age × Billing Amount`
- Calculated `Length_of_Stay`

---

## Model Training

- Random Forest (baseline)
- Random Forest + Hyperparameter tuning (RandomizedSearchCV)
- XGBoost and CatBoost (underperformed)

---

## Evaluation

**Final Confusion Matrix:**  
![Confusion Matrix](https://github.com/chazbrahma/Health_Classification_model/blob/main/Plots/confusion_matrix_rf_randomized%20(1).png?raw=true)

**Top Feature Importances:**  
![Feature Importance](https://github.com/chazbrahma/Health_Classification_model/blob/main/Plots/feature_importance_tuned_rf%20(1).png?raw=true)

---

## Final Performance

- Accuracy: **~41.5%**
- Top 4 features:
  - Billing Amount
  - Age × Billing
  - Age
  - Length of Stay

---


---

## Key Learnings

- Strong signal in billing-related features
- Random Forest was most robust model
- Inconclusive class remained hardest to predict

---

## Future Improvements

- Try ensembling (VotingClassifier)
- Use stratified K-Fold for more robust validation
- Normalize skewed features (e.g. log-billing)

---

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/Healthcare_Project.ipynb

##  License

For educational and demonstration purposes only.



