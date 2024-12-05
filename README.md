# Credit-Score-Classification-Dataset-Project-Multiclass
### Note: This project is organized into multiple branches on GitHub, each correlating with a different portion of the assignment. To explore specific sections, please select the branch that corresponds to the task or analysis you are interested in. 

In this project, I developed a machine learning system designed to automate the process of classifying individuals into specific credit score brackets based on their financial and credit-related data. The goal was to streamline and enhance the efficiency of credit evaluation for a global finance company, reducing the reliance on manual processes that could be time-consuming and error-prone. The system aimed to accurately categorize customers into predefined credit score ranges (e.g., Good, Standard, Poor) to aid in risk assessment, loan approvals, and personalized financial product offerings. By leveraging advanced statistical analysis and machine learning models, the project sought to ensure better decision-making, improve risk management, and optimize the company's ability to handle large volumes of customer data. 

Throughout the project, I focused on improving model accuracy, interpretability, and scalability by systematically exploring the dataset, conducting thorough correlation analysis, selecting relevant features, and fine-tuning the model for optimal performance. The outcome was an automated system capable of efficiently predicting the creditworthiness of customers, offering significant improvements in both operational efficiency and the quality of financial decision-making.

# Assignment Outcome: Comparison of Baseline Model vs Final Model
---
| Model                                         | Set     | Accuracy |
|-----------------------------------------------|---------|----------|
| **BASELINE MODEL - All Attributes - 10 Max Depth** | Training | 0.8176   |
| **BASELINE MODEL - All Attributes - 10 Max Depth** | Test     | 0.8113   |
| **FINAL MODEL - BINNED - Selected Attributes - 18 Max Depth**  | Training | 0.9103   |
| **FINAL MODEL - BINNED - Selected Attributes - 18 Max Depth**  | Test     | 0.8487   |

| Model                                                              | Set     | Accuracy | Sensitivity | Specificity | Precision |
|--------------------------------------------------------------------|---------|----------|-------------|-------------|-----------|
| **BASELINE MODEL - Good - Training Set - All Attributes - 10 Max Depth** | Training |          | 0.6989      | 0.8969      | 0.5953    |
| **BASELINE MODEL - Good - Test Set - All Attributes - 10 Max Depth**    | Test     |          | 0.6904      | 0.8926      | 0.5823    |
| **FINAL MODEL - BINNED - Good - Training Set - Selected Attributes - 18 Max Depth**  | Training |          | 0.9337      | 0.9281      | 0.8666    |
| **FINAL MODEL - BINNED - Good - Test Set - Selected Attributes - 18 Max Depth**      | Test     |          | 0.8656      | 0.8913      | 0.7992    |

| Model                                                              | Set     | Accuracy | Sensitivity | Specificity | Precision |
|--------------------------------------------------------------------|---------|----------|-------------|-------------|-----------|
| **BASELINE MODEL - Poor - Training Set - All Attributes - 10 Max Depth** | Training |          | 0.7242      | 0.8875      | 0.7244    |
| **BASELINE MODEL - Poor - Test Set - All Attributes - 10 Max Depth**    | Test     |          | 0.7203      | 0.8821      | 0.7139    |
| **FINAL MODEL - BINNED - Poor - Training Set - Selected Attributes - 18 Max Depth**  | Training |          | 0.9186      | 0.9260      | 0.8613    |
| **FINAL MODEL - BINNED - Poor - Test Set - Selected Attributes - 18 Max Depth**      | Test     |          | 0.8381      | 0.8807      | 0.7784    |

| Model                                                                 | Set     | Accuracy | Sensitivity | Specificity | Precision |
|-----------------------------------------------------------------------|---------|----------|-------------|-------------|-----------|
| **BASELINE MODEL Standard - Training Set - All Attributes - 10 Max Depth** | Training |          | 0.7367      | 0.7671      | 0.7823    |
| **BASELINE MODEL - Standard - Test Set - All Attributes - 10 Max Depth**    | Test     |          | 0.7240      | 0.7628      | 0.7761    |
| **FINAL MODEL - BINNED - Standard - Training Set - Selected Attributes - 18 Max Depth** | Training |          | 0.7440      | 0.9440      | 0.8691    |
| **FINAL MODEL - BINNED - Standard - Test Set - Selected Attributes - 18 Max Depth**    | Test     |          | 0.6154      | 0.8875      | 0.7323    |


# Branches: 
## *>Branch 1<*
### Baseline Decision Tree Model: Initial Performance Analysis:
- A baseline decision tree model was created without balancing the class distribution, serving as a starting point for further model refinement.
- The model was trained with a maximum depth of 10 to avoid overfitting while maintaining interpretability.
### Data Preparation and Correlation Analysis (Using R):
- Explored and cleaned the dataset, handling missing values
- Calculated correlations between attributes using various statistical methods:
  - Numeric vs. Categorical: ANOVA and Eta-Squared
  - Numeric vs. Numeric: Pearson Correlation Coefficient
  - Categorical vs. Categorical: Phi Coefficient/Cramér’s V, supported by contingency tables
- Exported a comprehensive correlation analysis report, including p-values, significance, and interpretations, to assist in attribute selection
### All Attributes Model Development (Using SAS):
- Built and evaluated decision tree and random forest models using all attributes
- Metrics such as accuracy, sensitivity, specificity, and recall were stored in an Excel sheet for comparison
- Generated variable importance tables for both models (decision tree and random forest) to guide feature selection
## *>Branch 2<*
### Selected Attributes Model - Strategy/Logic:
- Developed a strategy to select attributes for the selected attributes model:
  - Assessed the relationship between each attribute and the target variable, using the insights from the comprehensive correlation analysis report generated in *Branch 1*
  - Prioritized attributes based on their rankings in the variable importance tables generated by the decision tree and random forest models
## *>Branch 3<*
### Selected Attributes Model Development:
- Created a decision tree model with selected attributes, achieving superior performance compared to the all-attributes model
### Hyperparameter Tuning:
- Experimented with different max depth values for decision trees
- Identified an optimal depth of 18, which improved the model's performance
### Feature Engineering:
- Binned numeric attributes using domain-specific logic, enhancing the performance of both the all-attributes and selected-attributes models
- Documented the binning logic for reproducibility
## *>Branch 4<*
### Insights Report on Selected Attributes:
- Created a detailed report highlighting the selected attributes and their significance in predicting credit scores
- Included insights derived from statistical correlation, variable importance, and feature engineering, offering interpretability and justification for attribute selection
### Scoring and Prediction:
- Developed SAS code to apply the selected-attributes decision tree model to new datasets, incorporating binning criteria for accurate predictions

