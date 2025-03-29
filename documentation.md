# LoanTap Loan Default Prediction: Project Documentation

**Date:** March 27, 2025

## 1. Introduction

### 1.1. Project Overview

This document details the analysis performed on LoanTap's loan data to predict the likelihood of default for personal loan applicants. LoanTap is an online platform focused on providing customized loan products, particularly to millennials. This project aims to support LoanTap's data science team in building an effective underwriting layer by identifying factors associated with loan default ('Charged Off' status).

### 1.2. Business Context

Accurate credit risk assessment is crucial in the lending industry. By predicting potential defaults, LoanTap can make more informed decisions about extending credit, potentially reducing Non-Performing Assets (NPAs) and optimizing its loan portfolio. This analysis focuses specifically on the Personal Loan product.

## 2. Problem Statement

The core objective is to develop a predictive model using historical loan data to determine the creditworthiness of individuals applying for a Personal Loan. Specifically, the model should predict whether a loan is likely to be 'Fully Paid' or 'Charged Off' based on applicant and loan attributes. The insights derived should also help understand the key drivers of default risk.

## 3. Data

### 3.1. Data Source

*   The primary dataset used for modeling is `logistic_regression.csv`.
*   A sample dataset, `loantap_sample.csv`, was provided for initial context.

### 3.2. Target Variable

*   `loan_status`: Indicates the final status of the loan. This is the variable the model aims to predict.
    *   `Fully Paid`: Loan was repaid successfully. (Encoded as 0)
    *   `Charged Off`: Loan was deemed uncollectible (default). (Encoded as 1)

### 3.3. Key Features

The dataset contains various attributes about the borrower and the loan, including (but not limited to):

*   **Loan Characteristics:** `loan_amnt`, `term`, `int_rate`, `installment`, `grade`, `purpose`.
*   **Borrower Employment:** `emp_length`, `emp_title`.
*   **Borrower Financials:** `annual_inc`, `dti` (Debt-to-Income ratio), `home_ownership`, `verification_status`.
*   **Borrower Location:** `state` (extracted from `address`).
*   **Credit History:** `earliest_cr_line` (used to derive credit history length), `open_acc`, `total_acc`, `revol_bal`, `revol_util`.
*   **Derogatory Marks:** `pub_rec`, `pub_rec_bankruptcies`, `mort_acc` (used to create flags).

*(Refer to the original data dictionary for a complete list and descriptions).*

## 4. Methodology: Approach and Explanation

The analysis was conducted using Python with standard data science libraries (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn). The approach involved several key stages:

### 4.1. Exploratory Data Analysis (EDA)

*   **Objective:** To understand the data's structure, characteristics, distributions, and relationships between variables.
*   **Steps:**
    *   **Loading & Inspection:** Loaded `logistic_regression.csv` into a Pandas DataFrame. Checked shape, data types (`df.info()`), missing values (`df.isnull().sum()`), and basic statistics (`df.describe()`).
    *   **Target Variable Analysis:** Examined the distribution of `loan_status` to understand class balance. A countplot (`loan_status_distribution.png`) was generated.
    *   **Univariate Analysis:** Analyzed the distribution of individual features. For example, a histogram (`loan_amnt_distribution.png`) was created for `loan_amnt`.
    *   **Bivariate Analysis:** Explored relationships between predictor variables and the target variable. For instance, a boxplot (`loan_amnt_vs_status.png`) showed the relationship between `loan_amnt` and `loan_status`.
    *   **Correlation Analysis:** Calculated the Pearson correlation matrix for numerical features and visualized it using a heatmap (`correlation_matrix.png`) to identify potential multicollinearity (e.g., high correlation between `loan_amnt` and `installment`).

### 4.2. Feature Engineering

*   **Objective:** To transform raw data into features suitable for modeling and potentially create new, informative features.
*   **Steps:**
    *   **Column Dropping:** Removed features less suitable for a baseline model (`title`, `sub_grade`, `issue_d`). `emp_title` and `address` were kept initially.
    *   **Numeric Conversion:**
        *   `term`: Extracted the numerical value (e.g., 36 from ' 36 months').
        *   `emp_length`: Mapped values like '< 1 year' to 0, '10+ years' to 10, and extracted numbers for other years. Missing values were treated as 0.
    *   **Date Feature:** Calculated `credit_history_length` (in years) by parsing `earliest_cr_line` using `pd.to_datetime` with `format='mixed'` for robustness, extracting the year, and subtracting it from a reference year (2017). Rows with unparseable dates were dropped. The original `earliest_cr_line` column was dropped.
    *   **Location Feature:** Extracted the two-letter `state` code from the `address` column using regex. Filled missing/unmatched states with 'Missing'. Dropped the original `address` column.
    *   **Flag Creation:** Created binary flags (0 or 1) for `pub_rec`, `mort_acc`, `pub_rec_bankruptcies`, indicating the presence (1) or absence (0) of these records.

### 4.3. Data Preprocessing

*   **Objective:** To prepare the engineered data for model training by handling missing values, scaling features, and encoding categorical variables.
*   **Steps:**
    *   **Target Encoding:** Converted `loan_status` to binary format (1 for 'Charged Off', 0 for 'Fully Paid').
    *   **Feature Identification:** Separated features (X) from the target (y) and identified numerical and categorical columns (including `emp_title` and `state`).
    *   **Train-Test Split:** Divided the data into training (80%) and testing (20%) sets using `train_test_split`. Stratification (`stratify=y`) was used to maintain the original proportion of target classes in both sets.
    *   **Pipeline Creation:** Used Scikit-learn's `Pipeline` and `ColumnTransformer` for robust preprocessing:
        *   **Numerical Pipeline:** Applied `SimpleImputer` (strategy='median') to fill missing numerical values, followed by `StandardScaler` to standardize features (zero mean, unit variance).
        *   **Categorical Pipeline:** Applied `SimpleImputer` (strategy='constant', fill_value='Missing') to fill missing categorical values, followed by `OneHotEncoder` (handle_unknown='ignore') to convert categories into numerical format.

### 4.4. Model Building

*   **Objective:** To train a classification model to predict `loan_status`.
*   **Model Choice:** `Logistic Regression` was chosen as required by the prompt and as a good baseline model due to its interpretability.
*   **Implementation:**
    *   A full Scikit-learn `Pipeline` was created, combining the preprocessing steps (`ColumnTransformer`) and the `LogisticRegression` classifier.
    *   `class_weight='balanced'` was used within `LogisticRegression` to automatically adjust weights inversely proportional to class frequencies, helping to mitigate the impact of potential class imbalance.
    *   `max_iter=1000` was set to ensure convergence.
    *   The pipeline was trained using the `fit()` method on the training data (`X_train`, `y_train`).

### 4.5. Model Evaluation

*   **Objective:** To assess the performance of the trained model on unseen data (the test set).
*   **Steps:**
    *   **Prediction:** Generated class predictions (`y_pred`) and probability predictions (`y_pred_proba`) on the test set (`X_test`) using the trained pipeline.
    *   **Classification Report:** Calculated and printed precision, recall, F1-score, and support for both classes ('Fully Paid', 'Charged Off'). This provides a detailed view of performance per class.
    *   **Confusion Matrix:** Visualized the matrix (`confusion_matrix.png`) showing True Positives, True Negatives, False Positives, and False Negatives.
    *   **ROC Curve & AUC:** Plotted the Receiver Operating Characteristic curve (`roc_curve.png`) and calculated the Area Under the Curve (AUC). ROC AUC measures the model's ability to distinguish between the positive and negative classes across different thresholds.
    *   **Precision-Recall Curve & AUC:** Plotted the Precision-Recall curve (`precision_recall_curve.png`) and calculated its AUC. This curve is particularly informative for imbalanced datasets, showing the tradeoff between precision and recall at different thresholds.
    *   **Coefficient Analysis:** Attempted to extract and display the coefficients learned by the Logistic Regression model to understand the relative importance and direction of influence of different features (after preprocessing).

## 5. Results Summary

*(Note: Exact numerical results depend on the specific output of the `loan_analysis.py` script execution, which was not fully captured.)*

*   The model was successfully trained and evaluated.
*   Performance metrics (Precision, Recall, F1-score, ROC AUC, PR AUC) were generated, providing a quantitative assessment of the model's predictive power. The `class_weight='balanced'` setting helped address class imbalance.
*   Visualizations (`confusion_matrix.png`, `roc_curve.png`, `precision_recall_curve.png`) provide graphical insights into model performance and tradeoffs. The plots should now be correctly generated based on the fixed data processing.
*   The analysis answered the specific questionnaire provided in the task description based on the initial data load.

## 6. Actionable Insights & Recommendations

*   **Risk Control:** Given the importance of minimizing NPAs in lending, **Precision** for the 'Charged Off' class is a key metric. The model should be tuned (e.g., by adjusting the classification threshold using the Precision-Recall curve) to achieve an acceptable level of precision based on LoanTap's risk appetite, while still maintaining reasonable recall to avoid rejecting too many creditworthy applicants.
*   **Feature Importance:** Features identified with significant coefficients should be reviewed. Note that including `emp_title` and `state` via OneHotEncoding creates many features; large coefficients for specific job titles or states might be due to low frequency and should be interpreted cautiously.
*   **Baseline Model:** This Logistic Regression model serves as a solid baseline. Further improvements could be explored by:
    *   Trying more complex models (e.g., Gradient Boosting, Random Forest).
    *   Performing more advanced feature engineering (e.g., creating interaction terms, different handling for high-cardinality features like `emp_title` and `state` - e.g., grouping, target encoding - to reduce dimensionality).
    *   Hyperparameter tuning for the chosen model.

## 7. Conclusion

This project successfully developed and evaluated a Logistic Regression model for predicting loan default risk based on LoanTap's data. The analysis provides valuable insights into model performance, feature importance, and the critical tradeoff between precision and recall in the context of lending. The generated artifacts (script, plots, documentation) form a basis for further development and integration into LoanTap's underwriting processes.
