# Credit-Score-Classification-Dataset-Project-Multiclass
In this project, I developed a machine learning system designed to automate the process of classifying individuals into specific credit score brackets based on their financial and credit-related data. The goal was to streamline and enhance the efficiency of credit evaluation for a global finance company, reducing the reliance on manual processes that could be time-consuming and error-prone. The system aimed to accurately categorize customers into predefined credit score ranges (e.g., Excellent, Good, Fair, Poor) to aid in risk assessment, loan approvals, and personalized financial product offerings. By leveraging advanced statistical analysis and machine learning models, the project sought to ensure better decision-making, improve risk management, and optimize the company's ability to handle large volumes of customer data. 

Throughout the project, I focused on improving model accuracy, interpretability, and scalability by systematically exploring the dataset, conducting thorough correlation analysis, selecting relevant features, and fine-tuning the model for optimal performance. The outcome was an automated system capable of efficiently predicting the creditworthiness of customers, offering significant improvements in both operational efficiency and the quality of financial decision-making.

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
