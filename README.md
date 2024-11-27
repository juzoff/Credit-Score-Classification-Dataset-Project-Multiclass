# Credit-Score-Classification-Dataset-Project-Multiclass
### Note: This project is organized into multiple branches on GitHub, each correlating with a different portion of the assignment. To explore specific sections, please select the branch that corresponds to the task or analysis you are interested in. 

In this project, I developed a machine learning system designed to automate the process of classifying individuals into specific credit score brackets based on their financial and credit-related data. The goal was to streamline and enhance the efficiency of credit evaluation for a global finance company, reducing the reliance on manual processes that could be time-consuming and error-prone. The system aimed to accurately categorize customers into predefined credit score ranges (e.g., Good, Standard, Poor) to aid in risk assessment, loan approvals, and personalized financial product offerings. By leveraging advanced statistical analysis and machine learning models, the project sought to ensure better decision-making, improve risk management, and optimize the company's ability to handle large volumes of customer data. 

Throughout the project, I focused on improving model accuracy, interpretability, and scalability by systematically exploring the dataset, conducting thorough correlation analysis, selecting relevant features, and fine-tuning the model for optimal performance. The outcome was an automated system capable of efficiently predicting the creditworthiness of customers, offering significant improvements in both operational efficiency and the quality of financial decision-making.

# Branches: 
## *>Branch 1<*
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
- Generated variable importance tables for both models (decision tree and random forest) to guide feature selection.
## *>Branch 2<*
### Selected Attributes Strategy/Logic:
- Developed a strategy to select attributes based on:
  - Assessed the relationship between each attribute and the target variable, using the insights from the comprehensive correlation analysis report generated in *Branch 1*.
  - Prioritized attributes based on their rankings in the variable importance tables generated by the decision tree and random forest models.
- Designed a decision tree model using the selected attributes, aiming to achieve better performance than the model built with all attributes.
## *>Branch 3<*
### Selected Attributes Model Deployment:
- Created a decision tree model with selected attributes, achieving superior performance compared to the all-attributes model.
### Hyperparameter Tuning:
- Experimented with different max depth values for decision trees.
- Identified an optimal depth of 18, which improved the model's performance.
### Feature Engineering:
- Binned numeric attributes using domain-specific logic, enhancing the performance of both the all-attributes and selected-attributes models.
- Documented the binning logic for reproducibility.
## *>Branch 4<*
### Insights Report on Selected Attributes:
- Created a detailed report highlighting the selected attributes and their significance in predicting credit scores.
- Included insights derived from statistical correlation, variable importance, and feature engineering, offering interpretability and justification for attribute selection.
### Scoring and Prediction:
- Developed SAS code to apply the selected-attributes decision tree model to new datasets, incorporating binning criteria for accurate predictions.
