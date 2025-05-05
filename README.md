# Healthcare ML Classification Project

This project applies machine learning to classify patients into three diagnostic categories — **Abnormal**, **Inconclusive**, and **Normal** — based on anonymized healthcare features such as age, gender, admission data, and billing amount.

---

## Problem Statement

In a simulated hospital dataset, our objective is to predict a patient’s test result class based on available features at admission.

### Target Classes:
- Abnormal
- Inconclusive
- Normal

### Key Features:
- Age
- Gender
- Blood Type
- Medical Condition
- Admission Type
- Billing Amount
- Length of Stay
- Age_Billing (Age × Billing)

---

## Workflow Overview

### 1. Data Cleaning
- Dropped irrelevant columns: Name, Room Number, Insurance Provider, Doctor, Hospital, Medication.
- Transformed date fields to calculate `Length_of_Stay`.
- Created a composite feature: `Age_Billing`.

### 2. Exploratory Data Analysis (EDA)

**Test Results Distribution:**
![Test Results](figures4/test_results_distribution.png)

**Age Distribution:**
![Age Histogram](figures4/age_distribution.png)

**Medical Condition Frequency:**
![Medical Condition](figures4/medical_condition_distribution.png)

> These plots confirmed a balanced target variable and uniform spread across features.

---

## Feature Engineering
- One-hot encoded categorical variables.
- Engineered:
  - `Length_of_Stay` from admission/discharge dates
  - `Age_Billing` = Age × Billing Amount

---

## Modelling Approach

### Models Tried:
- Random Forest (Baseline + Tuned)
- XGBoost
- CatBoost

Final model selected: **Tuned Random Forest (via RandomizedSearchCV)**  
→ Balanced performance, interpretability, and efficient training.

---

## Evaluation

**Final Confusion Matrix:**
![Confusion Matrix](figures4/confusion_matrix_rf_randomized.png)

**Top 20 Feature Importances:**
![Feature Importances](figures4/feature_importance_tuned_rf.png)

### Final Accuracy: **~41.5%**
- Abnormal & Normal were better predicted than Inconclusive
- Most influential variables:
  - Billing Amount
  - Age_Billing
  - Age
  - Length of Stay

---


---

## Learnings

- Derived features (like Age × Billing) were very helpful
- Feature importance helped prune irrelevant data
- Random Forest was more stable than XGBoost/CatBoost in this setup

---

##  Future Work

- Try ensembling (Voting or Stacking)
- Use Stratified K-Fold CV
- Incorporate more time-based features or medical severity indices
- Deploy via API or streamlit dashboard

---

## To Run the Project

```bash
pip install -r requirements.txt
jupyter notebook notebooks/Healthcare_Project.ipynb

