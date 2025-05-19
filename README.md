# Loan Default Prediction - EDA & Modeling

This project focuses on predicting loan default outcomes using data from three different sources: account details, enquiry details, and previous flags. Performed Exploratory Data Analysis (EDA) and trained machine learning models (Random Forest & XGBoost) to classify loans as good or bad.

---

## Datasets Used

1. **Flag Data**
   - 'uid', 'NAME_CONTRACT_TYPE', 'TARGET'
2. **Accounts Data**
   - 'uid', 'credit_type', 'loan_amount', 'amount_overdue', 'open_date', 'closed_date', 'payment_hist_string'
3. **Enquiry Data**
   - 'uid', 'enquiry_type', 'enquiry_amt', 'enquiry_date'

---

## Exploratory Data Analysis (EDA)

- Filled missing values and handled date formatting
- Merged the datasets using 'uid', handling data for duplicate IDs by merging data into lists
- Performed univariate and bivariate analysis to understand trends in defaults
- Visualized correlations, overdue patterns, and enquiry behavior of users

---

## Model Training

### Models Used:
- **XGBoost**
- **XGBoost with Grid Search**

### Process:
- Extracted min, max, avg, etc features from numerical columns (e.g. 'loan_amount', 'amount_overdue', 'payment_hist_string', etc.)
- Using dunmmies and one-hot encoded categorical columns (eg. 'credit_type', 'enquiry_type', etc)
- Selected Numerical and Binary Feature from merged dataset
- Train-test split and model evaluation using accuracy, ROC-AUC-Score, F1 score
- Feature importance analysis to understand significant features in predicting of loan default

---

## Possible Future Work and Improvements:
- Domain Knowledge for better preprocessing steps
- Additional statistical values for numerical columns - variance, std dev, running average, quantiles, IQR
- Datetime based features - Year, Month, Quarter, Day, Day of Week, isWeekend
- Better resampling for bias reduction

---
