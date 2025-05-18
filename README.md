# LoanTap Loan Default Prediction Analysis

## 1. Introduction

This project analyzes loan data from LoanTap, an online platform providing customized loan products. The primary goal is to build a model to predict the likelihood of a borrower defaulting on a Personal Loan, aiding LoanTap's data science team in developing an underwriting layer for determining creditworthiness.

## 2. Problem Statement

Given a set of attributes for an individual applying for a Personal Loan, the objective is to:
1.  Determine if a credit line should be extended by predicting the loan status ('Fully Paid' or 'Charged Off').
2.  Provide insights that can inform risk assessment and potentially influence repayment term recommendations.

## 3. Dataset

*   **Primary Data:** `logistic_regression.csv`
*   **Sample Data:** `loantap_sample.csv` (Used for context, not model training)
*   **Target Variable:** `loan_status` ('Fully Paid' or 'Charged Off')
*   **Key Features (Examples):** `loan_amnt`, `term`, `int_rate`, `installment`, `grade`, `emp_title`, `emp_length`, `home_ownership`, `annual_inc`, `verification_status`, `dti`, `earliest_cr_line`, `open_acc`, `pub_rec`, `revol_bal`, `revol_util`, `total_acc`, `mort_acc`, `pub_rec_bankruptcies`, `state` (extracted). (Refer to the data dictionary in the initial prompt for full details).

## 4. Methodology

The analysis follows these steps:

1.  **Exploratory Data Analysis (EDA):**
    *   Loaded and inspected the dataset (`logistic_regression.csv`).
    *   Analyzed target variable distribution.
    *   Performed univariate analysis (e.g., distribution of `loan_amnt`).
    *   Performed bivariate analysis (e.g., `loan_amnt` vs. `loan_status`).
    *   Calculated and visualized correlations between numerical features.
2.  **Feature Engineering:**
    *   Dropped columns less suitable for a baseline model (`title`, `sub_grade`, `issue_d`). `emp_title` and `address` were kept initially.
    *   Converted `term` (e.g., ' 36 months') to numeric (e.g., 36).
    *   Converted `emp_length` (e.g., '10+ years', '< 1 year') to numeric (0-10).
    *   Calculated `credit_history_length` in years from `earliest_cr_line` (using robust date parsing).
    *   Extracted `state` from `address` using regex, then dropped `address`.
    *   Created binary flags for `pub_rec`, `mort_acc`, `pub_rec_bankruptcies` (1 if > 0, else 0).
3.  **Data Preprocessing:**
    *   Encoded the target variable `loan_status` (Charged Off: 1, Fully Paid: 0).
    *   Identified numerical and categorical features (including `emp_title` and `state`).
    *   Split data into training (80%) and testing (20%) sets, stratified by the target variable.
    *   Created preprocessing pipelines:
        *   Numerical: Median imputation + `StandardScaler`.
        *   Categorical: Constant imputation ('Missing') + `OneHotEncoder`.
4.  **Model Building:**
    *   Implemented a `Logistic Regression` model within a scikit-learn pipeline, incorporating the preprocessing steps.
    *   Used `class_weight='balanced'` to handle potential class imbalance.
5.  **Model Evaluation:**
    *   Generated predictions on the test set.
    *   Calculated and displayed the Classification Report (Precision, Recall, F1-score).
    *   Plotted and saved the Confusion Matrix (`confusion_matrix.png`).
    *   Plotted and saved the ROC Curve, calculating AUC (`roc_curve.png`).
    *   Plotted and saved the Precision-Recall Curve, calculating AUC (`precision_recall_curve.png`).
    *   Analyzed model coefficients to infer feature importance.

## 5. Setup & Usage

**Requirements:**

*   Python 3.x
*   Libraries:
    *   pandas
    *   numpy
    *   matplotlib
    *   seaborn
    *   scikit-learn

Install requirements using pip:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```
or (if pip is associated with a different Python version):
```bash
py -m pip install pandas numpy matplotlib seaborn scikit-learn
```

**Running the Analysis:**

Execute the Python script from your terminal in the project directory:
```bash
python loan_analysis.py
```
or (if `python` is not in PATH on Windows):
```bash
py loan_analysis.py
```
The script will print analysis steps, results, and save output plots (`.png` files) to the current directory.

## 6. Results

*   The script successfully loads the data, performs EDA, preprocesses features (including `emp_title`), trains a Logistic Regression model, and evaluates it.
*   Key evaluation metrics (Precision, Recall, F1-score, ROC AUC, PR AUC) are printed in the console output. (See latest execution output for values).
*   Visualizations are saved as PNG files:
    *   `loan_status_distribution.png`
    *   `loan_amnt_distribution.png`
    *   `loan_amnt_vs_status.png`
    *   `correlation_matrix.png`
    *   `confusion_matrix.png`
    *   `roc_curve.png`
    *   `precision_recall_curve.png`
*   **Answers to Questionnaire (from initial data load):**
    1.  **% Fully Paid:** 80.39%
    2.  **Corr(loan\_amnt, installment):** 0.9539
    3.  **Majority Home Ownership:** MORTGAGE
    4.  **Grade 'A' Likelihood:** True (0.94 vs 0.80)
    5.  **Top Job Titles:** ['Missing', 'Teacher'] (Based on frequency)
    6.  **Primary Metric:** Precision recommended to control NPAs, balanced with Recall.
    7.  **Precision-Recall Gap:** Highlights tradeoff between risk (NPAs) and opportunity (revenue).
    8.  **Influential Features:** See coefficient analysis in script output (influenced by `emp_title` and `state`).
    9.  **Geographical Effect:** Yes, the model includes `state`. Significance depends on coefficients (check script output).

## 7. Insights & Recommendations

*   The Logistic Regression model provides a baseline for predicting loan defaults. Including `emp_title` and `state` adds granularity but significantly increases feature dimensionality via OneHotEncoding.
*   Controlling Non-Performing Assets (NPAs) is critical. Focusing on **Precision** is recommended, potentially by adjusting the classification threshold.
*   Features identified by the model coefficients as strongly associated with default risk should be closely monitored. Note that individual job titles or states might have large coefficients due to low frequency combined with high/low default rates in small samples; interpret with caution.
*   Further improvements could involve exploring more complex models, dimensionality reduction techniques for high-cardinality features like `emp_title` and `state` (e.g., grouping, target encoding), and hyperparameter tuning.

## 8. Files

*   `loan_analysis.py`: Main Python script containing the analysis code.
*   `logistic_regression.csv`: The primary dataset used for training and evaluation.
*   `loantap_sample.csv`: Sample data provided for context.
*   `README.md`: This documentation file.
*   `*.png`: Output plots generated by the analysis script.
*   `documentation.md`: Detailed project documentation.

# Build and run Docker container locally
docker build -t loantap-api .
docker run -p 8080:8080 loantap-api

# Test the API
curl -X POST "http://localhost:8080/predict" \
     -H "Content-Type: application/json" \
     -d '{"loan_amnt": 10000, "term": "36 months", ...}'
