# Branch 1

## Data Preparation and Correlation Analysis (Using R):
### > FILE: 
    - Credit_Score - Data Preperation - Correlation Strength.Rmd
    - Credit_Score - Data Preperation - Correlation Strength.pdf (Knitted)
    - Credit_Score_Multiclass_CORRELATIONAL ANALYSIS.csv
    - Balancing Class Attribute.Rmd
    - Balancing Class Attribute.pdf (Knitted)
    - balanced_data_creditscore.csv
#### Highlights:
- Explored and cleaned the dataset, handling/checking missing values.
- Calculated correlations between attributes using various statistical methods:
  - Numeric vs. Categorical: ANOVA and Eta-Squared.
  - Numeric vs. Numeric: Pearson Correlation Coefficient.
  - Categorical vs. Categorical: Phi Coefficient/Cramér’s V, supported by contingency tables.
- Exported a comprehensive correlation analysis report via a csv file, including p-values, significance, and interpretations, to assist in attribute selection.
- In a separate R file (*Balancing Class Attribute.Rmd*), analyzed the distribution of the class attribute to identify imbalances.
  - The dataset showed a significant imbalance in the distribution of credit score classes. Specifically, the "Good" class had the lowest representation with 17,828 instances, while the "Poor" and "Standard" classes had much higher counts of 28,998 and 53,174 instances, respectively. This imbalance could lead to biased model predictions, where the model might perform well on the majority classes ("Poor" and "Standard") but struggle to accurately predict the minority class ("Good").
  - To address this, I applied undersampling, a technique where I reduced the number of instances in the overrepresented classes ("Poor" and "Standard") to match the count of the underrepresented "Good" class. After undersampling, all three classes had an equal count of 17,828 instances. This balanced dataset ensured that the model had an equal opportunity to learn patterns from each class, leading to fairer and more generalized predictions across all credit score categories.
  - Exported balanced dataset to csv file (*balanced_data_creditscore.csv*)

## All Attributes Model Development (Using SAS):
### > FILE: 
    - All Attributes Model Development.sas
- Inserted balanced CSV dataset (*balanced_data_creditscore*) to be used in the decision tree and random forest models
  - Double checked balanced class distribution
- Built and evaluated decision tree and random forest models using all attributes.
- Metrics such as accuracy, sensitivity, specificity, and recall for decision tree training and test set were stored in an Excel sheet for comparison.

  - Training set results:
![Training all](https://github.com/user-attachments/assets/c820c532-4bc3-4853-95a1-65fcc841a697)

  - Test set results:
![test all](https://github.com/user-attachments/assets/5000743b-99dd-462b-851b-039563c5a6d6)

 #### The following tables were stored to assist in attribute selection in the second branch of this GitHub Repository: 
- All attributes decision tree variable importance table identified the most influential attributes based on the reduction in impurity at splits:
  ![variable importance decision tree](https://github.com/user-attachments/assets/4f1a4b2f-93a4-46e3-850f-c38ec2c89656)
- Random forest variable importance table ranked attributes based on the mean decrease in accuracy or Gini index across the ensemble of trees (second image is attributes sorted by Gini):
  ![variable importance random forest](https://github.com/user-attachments/assets/128204ea-ab1a-4f34-b409-32df7537623b)
![variable importance random forest2](https://github.com/user-attachments/assets/5094b9ec-e086-4ec2-b34a-fc5c375428c5)


