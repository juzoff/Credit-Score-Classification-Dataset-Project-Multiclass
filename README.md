# Credit-Score-Classification-Dataset-Project-Multiclass

---

## PROJECT GOAL: Develop robust multiclass classification models (Decision Trees, Random Forests, Multinomial Logistic Regression, KNN) that effectively balances risk mitigation and customer acquisition by accurately predicting credit scores across three categories: "Poor," "Standard," and "Good."

---

### Step 1: Research Objectives ###
---
1. Research Objective # 1: Risk Mitigation: Minimizing the Misclassification of Poor Credit Scores
* Goal: Reduce the risk of classifying "Poor" credit scores as "Standard" or "Good" to avoid approving risky individuals.
* Key Metric to Focus On: Recall for the "Poor" class → Ensures that most truly poor credit scores are identified correctly.

2. Research Objective # 2: Customer Acquisition: Focused on Good and Standard Credit Scores
* Goal: Maximize marketing efforts by accurately identifying "Good" and "Standard" credit scores for customer acquisition.
* Key Metric to Focus On: Precision for Good and Standard Classes → Ensures that most individuals classified as "Good" or "Standard" are truly in those categories, minimizing wasted marketing efforts.

3. Research Objective # 3 (BONUS: SEE BOTTOM OF PAGE): Fairness in Credit Score Classification: Ensuring Equal Recall Across Demographic Groups
* Goal: Ensure that individuals from different demographic groups (e.g., gender, age, income) receive equal treatment in terms of credit score classification, particularly for the "Poor" class. This will help mitigate bias and ensure fairness in the model's performance.
* Key Metric to Focus On: Recall for "Poor" Credit Scores Across Demographic Groups → Ensures that the model identifies "Poor" credit scores equally well across different groups (e.g., gender, age, income).

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

### Step 3: Decision Tree ###
---
1. Data Loading and Initial Exploration:
- Loaded the cleaned dataset with 10,000 rows and 23 columns (10 float64, 6 int64, 7 object).
- Confirmed no missing values and checked the distribution of Credit_Score: Standard (5317), Poor (2900), Good (1783).

2. Model 1: Decision Tree with Balancing - All Features:
- Preprocessing: Encoded categorical features using one-hot encoding (pd.get_dummies).
- Train-Test Split: Split data into 70% training (7000 rows) and 30% testing (3000 rows).
- Balancing:
  - Undersampled Standard to match Poor (2046) using RandomUnderSampler.
  - Oversampled Good to 2046 using SMOTE, resulting in balanced training data (2046 each).
- Hyperparameter Tuning: Used GridSearchCV with parameters (max_depth, min_samples_split, min_samples_leaf, criterion='gini') to find the best settings (e.g., max_depth=10).
- Training and Evaluation: Trained a Decision Tree with the best parameters and evaluated using accuracy (0.6543), precision, recall, specificity per class, and a confusion matrix.
- Results: Recall for Poor (0.60), Precision for Good (0.74), balanced performance across classes.

3. Model 2: Decision Tree without Balancing - All Features:
- Preprocessing: Same encoding and train-test split as Model 1.
- Balancing: No balancing applied; used original class distribution (Standard: 3723, Poor: 2046, Good: 1231 in training).
- Hyperparameter Tuning: GridSearchCV identified best parameters (e.g., max_depth=10).
- Training and Evaluation: Trained and evaluated, achieving higher accuracy (0.6713), with better recall for Good (0.75) but lower recall for Poor (0.55).
- Results: Favored majority class (Standard/Good) due to imbalance.

4. Model 3: Decision Tree with Balancing - Selected Features:
- Preprocessing: Same encoding and split as above.
- Balancing: Applied undersampling (Standard to 2046) and SMOTE (Good to 2046), as in Model 1.
- Feature Selection: Used RFECV with a Decision Tree to select the top 10 features (e.g., Age, Annual_Income, Num_Bank_Accounts).
- Hyperparameter Tuning: GridSearchCV found best parameters (e.g., max_depth=10, min_samples_leaf=8).
- Training and Evaluation: Trained on selected features, resulting in lower accuracy (0.5727) but higher recall for Poor (0.75), though recall for Good dropped (0.46).
- Results: Improved Poor class detection at the cost of overall accuracy.

5. Model 4: Decision Tree without Balancing - Selected Features:
- Preprocessing: Same encoding, split, and no balancing as Model 2.
- Feature Selection: RFECV selected the same top 10 features as Model 3.
- Hyperparameter Tuning: Best parameters via GridSearchCV (e.g., max_depth=10).
- Training and Evaluation: Accuracy (0.6370), balanced recall across classes (Poor: 0.56, Standard: 0.56, Good: 0.70).
- Results: Moderate performance, less biased toward majority class than Model 2.

6. Evaluation Metrics:
- For each model, computed accuracy, recall, precision, specificity per class, and confusion matrices. Specificity calculated as TN/(TN+FP).
7. Visualization:
Created a bar plot comparing accuracy, specificity, recall, and precision across all four models using matplotlib and pandas, with a large figure size (20x15) for readability.

### Step 4: Random Forest ###
---
1. Data Loading and Initial Exploration:
- Loaded the dataset with 10,000 rows and 23 columns (10 float64, 6 int64, 7 object).
- Confirmed no missing values and checked Credit_Score distribution: Standard (5317), Poor (2900), Good (1783).

2. Model 1: Random Forest with Balancing - All Features:
- Preprocessing: Encoded categorical features using one-hot encoding (pd.get_dummies).
- Train-Test Split: Split into 70% training (7000 rows) and 30% testing (3000 rows).
- Balancing:
  - Undersampled Standard to match Poor (2046) using RandomUnderSampler.
  - Oversampled Good to 2046 using SMOTE, balancing training data (2046 each).
- Hyperparameter Tuning: Used GridSearchCV with parameters (n_estimators, max_depth, min_samples_split, min_samples_leaf, criterion='gini') to find the best settings (e.g., n_estimators=150, max_depth=None).
- Training and Evaluation: Trained a Random Forest with the best parameters, achieving accuracy (0.6563), with high recall for Poor (0.76) and precision for Good (0.80).
- Results: Strong Poor class detection, balanced performance.

3. Model 2: Random Forest without Balancing - All Features:
- Preprocessing: Same encoding and split as Model 1.
- Balancing: No balancing; used original distribution (Standard: 3723, Poor: 2046, Good: 1231 in training).
- Hyperparameter Tuning: GridSearchCV identified best parameters (e.g., n_estimators=100, max_depth=None).
- Training and Evaluation: Accuracy (0.6693), higher recall for Good (0.74), lower for Poor (0.49).
- Results: Favored majority class (Good/Standard) due to imbalance.

4. Model 3: Random Forest with Balancing - Selected Features:
- Preprocessing: Same encoding and split.
- Balancing: Applied undersampling (Standard to 2046) and SMOTE (Good to 2046), as in Model 1.
- Feature Selection: Used a base Random Forest (n_estimators=50) to compute feature importances and selected the top 10 features (e.g., Outstanding_Debt, Interest_Rate).
- Hyperparameter Tuning: Used RandomizedSearchCV (fewer iterations than GridSearch) with a smaller parameter grid, finding best settings (e.g., n_estimators=150, max_depth=20).
- Training and Evaluation: Accuracy (0.6767), high recall for Poor (0.75) and Standard (0.75), precision for Good (0.80).
- Results: Best overall balanced performance with fewer features.

5. Model 4: Random Forest without Balancing - Selected Features:
- Preprocessing: Same encoding and split, no balancing (original distribution).
- Feature Selection: Same method as Model 3, selecting top 10 features (e.g., Outstanding_Debt, Delay_from_due_date).
- Hyperparameter Tuning: RandomizedSearchCV found best parameters (e.g., n_estimators=150, max_depth=None).
- Training and Evaluation: Highest accuracy (0.6773), best recall for Good (0.77), lowest for Poor (0.45).
- Results: Strong Good class performance, weaker Poor detection.

6. Evaluation Metrics:
- For each model, computed accuracy, recall, precision, specificity per class, and confusion matrices. Specificity calculated as TN/(TN+FP).
7. Visualization:
Created a bar plot comparing accuracy, specificity, recall, and precision across all four models using matplotlib and pandas, with a large figure size (20x15) for readability.

### Step 5: Multinomial Logistic Regression ###
---
1. Data Loading and Initial Exploration:
- Loaded the dataset with 10,000 rows and 23 columns (10 float64, 6 int64, 7 object).
- Confirmed no missing values and checked Credit_Score distribution: Standard (5317), Poor (2900), Good (1783).
2. Model 1: Logistic Regression with Balancing - All Features:
- Preprocessing: Encoded categorical features with one-hot encoding (pd.get_dummies) and standardized numeric features using StandardScaler.
- Train-Test Split: Split into 70% training (7000 rows) and 30% testing (3000 rows).
- Balancing: Undersampled Standard to match Poor (2046) with RandomUnderSampler, then oversampled Good to 2046 with SMOTE, balancing training data (2046 each).
- Hyperparameter Tuning: Used RandomizedSearchCV with parameters (C, solver='lbfgs', max_iter) to find best settings (e.g., C=1.934, max_iter=200).
- Training and Evaluation: Trained the model, achieving accuracy (0.6513), recall for Poor (0.65), and precision for Good (0.75). Noted convergence warnings suggesting more iterations or better scaling.
- Results: Balanced performance with decent Poor recall.
3. Model 2: Logistic Regression without Balancing - All Features:
- Preprocessing: Same encoding and standardization as Model 1.
- Balancing: No balancing; used original distribution (Standard: 3723, Poor: 2046, Good: 1231 in training).
- Hyperparameter Tuning: RandomizedSearchCV identified best parameters (e.g., C=1.934, max_iter=200).
- Training and Evaluation: Accuracy (0.6520), higher recall for Good (0.75), lower for Poor (0.50). Convergence warnings persisted.
- Results: Favored majority class (Good), weaker Poor detection.
4. Model 3: Logistic Regression with Balancing - Selected Features:
- Preprocessing: Same encoding and standardization.
- Balancing: Same undersampling and SMOTE as Model 1.
- Feature Selection: Used a Random Forest (n_estimators=50) to select the top 10 features based on importance (e.g., Annual_Income, Num_Credit_Card).
- Hyperparameter Tuning: RandomizedSearchCV found best parameters (e.g., C=3.845, max_iter=100).
- Training and Evaluation: Accuracy (0.5940), highest recall for Poor (0.74), lower recall for Good (0.51).
- Results: Strong Poor class detection, reduced overall accuracy.
5. Model 4: Logistic Regression without Balancing - Selected Features:
- Preprocessing: Same encoding and standardization, no balancing.
- Feature Selection: Same method as Model 3, selecting top 10 features (e.g., Monthly_Inhand_Salary, Outstanding_Debt).
- Hyperparameter Tuning: Best parameters from RandomizedSearchCV (e.g., C=3.845, max_iter=100).
- Training and Evaluation: Accuracy (0.6383), highest recall for Good (0.79), lowest for Poor (0.40).
- Results: Strong Good class performance, poor Poor recall.
6. Evaluation Metrics:
- For each model, computed accuracy, recall, precision, specificity per class, and confusion matrices. Specificity calculated as TN/(TN+FP).
7. Visualization:
Created a bar plot comparing accuracy, specificity, recall, and precision across all four models using matplotlib and pandas, with a large figure size (20x15) for readability.

### Step 6: KNN ###
---
1. Data Loading and Initial Exploration:
- Loaded the dataset with 10,000 rows and 23 columns (10 float64, 6 int64, 7 object).
- Confirmed no missing values and checked Credit_Score distribution: Standard (5317), Poor (2900), Good (1783).
2. Model 1: KNN with Balancing - All Features:
- Preprocessing: Encoded categorical features with one-hot encoding (pd.get_dummies) and standardized numeric features using StandardScaler.
- Train-Test Split: Split into 70% training (7000 rows) and 30% testing (3000 rows).
- Balancing: Undersampled Standard to match Poor (2046) with RandomUnderSampler, then oversampled Good to 2046 with SMOTE, balancing training data (2046 each).
- Hyperparameter Tuning: Used RandomizedSearchCV with parameters (n_neighbors, weights, metric) to find best settings (e.g., n_neighbors=6, metric='euclidean', weights='distance').
- Training and Evaluation: Trained the model, achieving accuracy (0.6287), recall for Poor (0.65), and precision for Good (0.75).
- Results: Balanced performance with decent Poor recall.
3. Model 2: KNN without Balancing - All Features:
- Preprocessing: Same encoding and standardization as Model 1.
- Balancing: No balancing; used original distribution (Standard: 3723, Poor: 2046, Good: 1231 in training).
- Hyperparameter Tuning: RandomizedSearchCV identified best parameters (e.g., n_neighbors=13, metric='minkowski', weights='uniform').
- Training and Evaluation: Accuracy (0.6563), higher recall for Good (0.74), lower for Poor (0.51).
- Results: Favored majority class (Good), weaker Poor detection.
4. Model 3: KNN with Balancing - Selected Features:
- Preprocessing: Same encoding and standardization.
- Balancing: Same undersampling and SMOTE as Model 1.
- Feature Selection: Used a Random Forest (n_estimators=50) to select the top 10 features based on importance (e.g., Annual_Income, Num_Credit_Card).
- Hyperparameter Tuning: RandomizedSearchCV found best parameters (e.g., n_neighbors=6, metric='euclidean', weights='distance').
- Training and Evaluation: Accuracy (0.6123), highest recall for Poor (0.70), lower recall for Good (0.53).
- Results: Strong Poor class detection, lowest overall accuracy.
5. Model 4: KNN without Balancing - Selected Features:
- Preprocessing: Same encoding and standardization, no balancing.
- Feature Selection: Same method as Model 3, selecting top 10 features (e.g., Monthly_Inhand_Salary, Outstanding_Debt).
- Hyperparameter Tuning: Best parameters from RandomizedSearchCV (e.g., n_neighbors=17, metric='minkowski', weights='uniform').
- Training and Evaluation: Highest accuracy (0.6593), highest recall for Good (0.75), lowest for Poor (0.46).
- Results: Strong Good class performance, poor Poor recall.
6. Evaluation Metrics:
- For each model, computed accuracy, recall, precision, specificity per class, and confusion matrices. Specificity calculated as TN/(TN+FP).
7. Visualization:
- Created a bar plot comparing accuracy, specificity, recall, and precision across all four models using matplotlib and pandas, with a large figure size (20x15) for readability.

---

## CHOOSING STRONGEST MODEL WHEN CONSIDERING RESEARCH OBJECTIVES - ANALYSIS OF MODELS
---
### Decision Trees:
![image](https://github.com/user-attachments/assets/f3918fe7-cadb-45bb-b235-9555a948f90e)

### Random Forests:
![image](https://github.com/user-attachments/assets/24cf0464-bfe5-480f-badc-15679e2807d7)

### Multinomial Logistic Regression:
![image](https://github.com/user-attachments/assets/95ac3e2b-9a39-42a5-8bd8-6df1aa7cd13f)

### KNN:
![image](https://github.com/user-attachments/assets/8927264a-ba4f-4bf1-9c65-fef4eb03f8e4)

---

### Research Objective # 1
- Strongest Model: Random Forest (With Balancing - All Features)
  - Poor Recall: 0.76 (highest across all models)
  - This model is the most effective at identifying risky individuals.
### Research Objective # 2
- Strongest Model (i): Random Forest (With Balancing - Selected Features)
  - Good Precision: 0.80 (highest across all models)
  - This model excels at accurately identifying Good credit scores, optimizing marketing efforts for high-value customers.
- Strongest Model (ii): Random Forest (Without Balancing - Selected Features)
  - Standard Precision: 0.70 (highest across all models)
  - This model is the most effective at correctly classifying Standard credit scores, reducing wasted marketing efforts on this group.

---

## CONCLUSION
---

The Credit-Score-Classification-Dataset-Project-Multiclass developed multiclass classification models to predict "Poor," "Standard," and "Good" credit scores, balancing risk mitigation and customer acquisition. Random Forests outperformed Decision Trees, Multinomial Logistic Regression, and KNN. For risk mitigation, Random Forest (With Balancing - All Features) achieved the highest Poor Recall (0.76), effectively identifying risky individuals. For customer acquisition, Random Forest (With Balancing - Selected Features) excelled with Good Precision (0.80), while Random Forest (Without Balancing - Selected Features) led with Standard Precision (0.70). 

### OBSERVATION: The balanced Random Forest with selected features (Poor Recall: 0.75, Good Precision: 0.80, Standard Precision: 0.64, Accuracy: 0.6767) offers the best trade-off for both objectives, making it the recommended model for robust credit score classification.

---

### (BONUS) Research Objective # 3: Fairness in Credit Score Classification: Ensuring Equal Recall Across Demographic Groups 
- Research Objective: The goal of this research is to ensure fairness in credit score classification by focusing on equal recall rates for individuals classified as "Poor" across different demographic groups (occupation, age, income). The aim is to mitigate bias in the model and ensure that all groups are treated equally in the prediction of creditworthiness.
- Key Focus: The primary metric for fairness is recall for "Poor" credit scores, which measures the model's ability to correctly identify individuals with poor credit scores. The disparity in recall rates across demographic groups will be assessed to identify potential bias in the model’s performance.
- Model Used: The Balanced Random Forest with selected features is used for this analysis. This model offers the best trade-off between different metrics (Poor Recall: 0.75, Good Precision: 0.80, Standard Precision: 0.64, Accuracy: 0.6767), making it the recommended model for achieving both fairness and robustness in credit score classification.

#### Analysis:
![image](https://github.com/user-attachments/assets/c0ddbcf9-0c65-4d71-a40f-cb0e4ebacac2)

![image](https://github.com/user-attachments/assets/3cd3a2bf-cd1a-4017-bafc-3ff6ff868059)

![image](https://github.com/user-attachments/assets/965eac3e-f32b-491b-b774-3f0ccf4661a8)

- Age Groups
  - Recall for "Poor" credit scores rises with younger age: Older Adults (0.47), Middle-Aged Adults (0.58), Young Adults (0.68), Younger Adults (0.73). The model struggles most with Older Adults (47% identified) and excels with Younger Adults (73%), a 26-point gap. This suggests bias against older individuals, possibly due to unrepresentative features or data, potentially disadvantaging them in credit access.

- Occupations
  - Recall varies widely across occupations, from 0.53 (Manager) to 0.82 (Scientist), a 29-point spread. High-recall roles like Scientist (0.82) and Developer (0.73) contrast with low-recall ones like Manager (0.53) and Journalist (0.56). This uneven performance hints at occupation-specific biases, possibly tied to income volatility, penalizing certain professions.

- Income Groups
  - Recall spans from 0.13 (High Income) to 0.78 (Low Income), a 65-point gap. The model rarely flags High Income poor risks (13%), likely due to data rarity, while excelling for Low Income (78%). This disparity suggests high-income individuals avoid scrutiny, while low-income ones face it more, risking socioeconomic inequity.


#### Conclusion:
- The Balanced Random Forest model, while robust (Poor Recall: 0.75, Accuracy: 0.6767), fails to ensure fairness in credit score classification. Recall disparities—26 points across age groups, 29 across occupations, and 65 across income groups—reveal significant bias, particularly disadvantaging older adults, certain professions, and low-income individuals. To meet the objective of equal recall across demographics, the model requires data rebalancing and fairness adjustments.


