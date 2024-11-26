# Credit-Score-Classification-Dataset-Project-Multiclass
In this project, I developed a machine learning system to classify individuals into credit score brackets using a large financial dataset. This project aimed to reduce manual effort and improve the efficiency of credit evaluation for a global finance company.

## Approach
### Data Preparation and Correlation Analysis (Using R):
- Explored and cleaned the dataset, handling missing values.
- Calculated correlations between attributes using various statistical methods:
  - Numeric vs. Categorical: ANOVA and Eta-Squared.
  - Numeric vs. Numeric: Pearson Correlation Coefficient.
  - Categorical vs. Categorical: Phi Coefficient/Cramér’s V, supported by contingency tables.
- Exported a comprehensive correlation analysis report, including p-values, significance, and interpretations, to assist in attribute selection.
### All Attributes Model Development (Using SAS):
- Built and evaluated decision tree and random forest models using all attributes.
- Metrics such as accuracy, sensitivity, specificity, and recall were stored in an Excel sheet for comparison.
- Generated variable importance tables for both models to guide feature selection.
### Selected Attributes Model Development:
- Developed a strategy to select attributes based on:
  - Correlation with the target variable (from R analysis).
  - Variable importance from decision trees and random forests.
- Created a decision tree model with selected attributes, achieving superior performance compared to the all-attributes model.
### Hyperparameter Tuning:
- Experimented with different max depth values for decision trees.
- Identified an optimal depth of 18, which improved the model's performance.
### Feature Engineering:
- Binned numeric attributes using domain-specific logic, enhancing the performance of both the all-attributes and selected-attributes models.
- Documented the binning logic for reproducibility.
### Insights Report on Selected Attributes:
- Created a detailed report highlighting the selected attributes and their significance in predicting credit scores.
- Included insights derived from statistical correlation, variable importance, and feature engineering, offering interpretability and justification for attribute selection.
### Scoring and Prediction:
- Developed SAS code to apply the selected-attributes decision tree model to new datasets, incorporating binning criteria for accurate predictions.
