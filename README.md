# Credit-Score-Classification-Dataset-Project-Multiclass

### PROJECT GOAL: Develop robust multiclass classification models that effectively balances risk mitigation and customer acquisition by accurately predicting credit scores across three categories: "Poor," "Standard," and "Good."

---

### Step 1: Research Objectives ###
---
1. Research Objective 1: Risk Mitigation: Minimizing the Misclassification of Poor Credit Scores
* Goal: Reduce the risk of classifying "Poor" credit scores as "Standard" or "Good" to avoid approving risky individuals.
* Key Metric to Focus On: Recall for the "Poor" class → Ensures that most truly poor credit scores are identified correctly.

2. Research Objective 2: Customer Acquisition: Focused on Good and Standard Credit Scores
* Goal: Maximize marketing efforts by accurately identifying "Good" and "Standard" credit scores for customer acquisition.
* Key Metric to Focus On: Precision for Good and Standard Classes → Ensures that most individuals classified as "Good" or "Standard" are truly in those categories, minimizing wasted marketing efforts.

### Step 2: Data Preprocessing ###
---
1. Data Loading and Initial Sampling:
- Loaded the dataset from a CSV file (credit_score.csv) using pandas.
- Reduced the dataset from 100,000 rows to 10,000 rows using stratified sampling with train_test_split to maintain the distribution of Credit_Score (Standard: 5317, Poor: 2900, Good: 1783).
2. Data Type Correction:
- Identified columns with incorrect data types (e.g., Age, Annual_Income, Num_of_Loan, etc., stored as objects).
- Converted these columns to numeric types using pd.to_numeric with errors='coerce' to handle invalid values, creating a new DataFrame data2.
3. Outlier Detection:
- Used the Interquartile Range (IQR) method to identify outliers in numeric columns (e.g., Age, Annual_Income, Num_of_Loan, etc.).
- Reported the number of outliers for each feature (e.g., 296 in Age, 268 in Annual_Income).
4. Handling Negative Values:
- Created data3 from data2 and replaced negative values in specific columns (Age, Num_Bank_Accounts, Num_of_Loan, Num_of_Delayed_Payment) with NaN to ensure data realism.
5. Outlier Removal:
- Applied the IQR method to replace extreme values with NaN in columns like Age (range: 14-56), Num_Bank_Accounts (0-10), Num_Credit_Card (0-11), Interest_Rate (1-34), Num_of_Loan (0-9), Num_of_Delayed_Payment (0-28), Num_Credit_Inquiries (0-17), and Total_EMI_per_month (0-356.98).
6. Column Removal:
- Created data5 from data4 and dropped columns deemed unnecessary or problematic: Credit_History_Age (all null), Name, SSN, ID, and Customer_ID.
7. Statistical Analysis:
- Conducted Anderson-Darling tests on continuous columns, rejecting normality for all (e.g., Age, Annual_Income).
- Performed Kruskal-Wallis tests on numeric columns against Credit_Score, finding significant differences across groups.
- Applied Chi-Square tests on categorical columns (Occupation, Type_of_Loan, etc.), identifying significant associations with Credit_Score except for Occupation.
8. Missing Value Imputation:
- Used KNN Imputation (k=5) on numeric columns after standard scaling, then reversed the scaling and rounded whole-number columns (e.g., Age, Num_Bank_Accounts) to integers.
- Replaced invalid categorical entries (e.g., '!@9#%8' in Payment_Behaviour, '_' in Credit_Mix, 'NM' in Payment_of_Min_Amount, '_______' in Occupation) with NaN.
- Imputed remaining missing categorical values using mode imputation (e.g., Type_of_Loan with "Credit-Builder Loan", Occupation with "Lawyer").
9. Final Dataset:
- Resulted in data_6 with no missing values across 23 columns, maintaining the original structure but with cleaned and imputed data.
#### This preprocessing pipeline addressed data type issues, outliers, negative values, missing data, and statistical validation, preparing the dataset for further analysis or modeling.

### Step 3: Decision Trees ###
---



