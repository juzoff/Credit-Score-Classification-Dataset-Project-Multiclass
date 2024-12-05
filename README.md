# Branch 1

## Baseline Decision Tree Model: Initial Performance Analysis:
### > FILE: 
    - Baseline Model.sas
#### Highlights:
- A baseline decision tree model was created without balancing the class distribution, serving as a starting point for further model refinement.
- The model was trained with a maximum depth of 10 to avoid overfitting while maintaining interpretability.
    - Performance Metrics:
        - Training Accuracy: 0.8019
        - Test Accuracy: 0.7458

## Data Preparation and Correlation Analysis (Using R):
### > FILE: 
    - Credit_Score - Data Preperation - Correlation Strength.Rmd
    - Credit_Score - Data Preperation - Correlation Strength.pdf (Knitted)
    - Credit_Score_Multiclass_CORRELATIONAL ANALYSIS.csv
    - Balancing Class Attribute.Rmd
    - Balancing Class Attribute.pdf (Knitted)
    - balanced_data_creditscore.csv
#### Highlights:
- Explored and cleaned the dataset, handling/checking missing values
- Calculated correlations between attributes using various statistical methods:
  - Numeric vs. Categorical: ANOVA and Eta-Squared
  - Numeric vs. Numeric: Pearson Correlation Coefficient
  - Categorical vs. Categorical: Phi Coefficient/Cramér’s V, supported by contingency tables
- Exported a comprehensive correlation analysis report via a csv file, including p-values, significance, and interpretations, to assist in attribute selection
- In a separate R file (*Balancing Class Attribute.Rmd*), analyzed the distribution of the class attribute to identify imbalances
  - The dataset showed a significant imbalance in the distribution of credit score classes. Specifically, the "Good" class had the lowest representation with 17,828 instances, while the "Poor" and "Standard" classes had much higher counts of 28,998 and 53,174 instances, respectively. This imbalance could lead to biased model predictions, where the model might perform well on the majority classes ("Poor" and "Standard") but struggle to accurately predict the minority class ("Good").
  - To address this, I applied undersampling, a technique where I reduced the number of instances in the overrepresented classes ("Poor" and "Standard") to match the count of the underrepresented "Good" class. After undersampling, all three classes had an equal count of 17,828 instances. This balanced dataset ensured that the model had an equal opportunity to learn patterns from each class, leading to fairer and more generalized predictions across all credit score categories.
  - Exported balanced dataset to csv file (*balanced_data_creditscore.csv*)

## All Attributes Model Development (Using SAS):
### > FILE: 
    - All Attributes Model Development.sas
    - All Attributes - Metrics and Visuals.xlsx
- Inserted balanced CSV dataset (*balanced_data_creditscore*) to be used in the decision tree and random forest models
  - Double checked balanced class distribution
- I initially started with a max depth of 10 for the decision tree model, which resulted in an accuracy of 0.8475 on the training set and 0.8254 on the test set. However, recognizing the importance of adjusting the max depth for larger datasets, I gradually increased it. After experimenting with different values, I settled on a max depth of 15, which provided a more appropriate starting benchmark for the model's performance
- Built and evaluated decision tree and random forest models using all attributes
  - Used a maxdepth of 15
- Metrics such as accuracy, sensitivity, specificity, and recall for decision tree training and test set were stored in an Excel sheet for comparison

  - Training set results:
![Training all](https://github.com/user-attachments/assets/c820c532-4bc3-4853-95a1-65fcc841a697)

  - Test set results:
![test all](https://github.com/user-attachments/assets/5000743b-99dd-462b-851b-039563c5a6d6)

![All Attributes - Accuracy](https://github.com/user-attachments/assets/5fe3d00f-34a1-4de8-bb93-1e25841154ac)
![All Attributes - Metrics and Visuals](https://github.com/user-attachments/assets/3baade3b-5e79-4cf8-a690-879a16b2a305)


 #### The following tables were stored to assist in attribute selection in the second branch of this GitHub Repository: 
- All attributes decision tree variable importance table identified the most influential attributes based on the reduction in impurity at splits:
  ![variable importance decision tree](https://github.com/user-attachments/assets/4f1a4b2f-93a4-46e3-850f-c38ec2c89656)
- Random forest variable importance table ranked attributes based on the mean decrease in accuracy or Gini index across the ensemble of trees (the image to the right are the attributes sorted by Gini from high to low):

  ![variable importance random forest](https://github.com/user-attachments/assets/128204ea-ab1a-4f34-b409-32df7537623b)
![variable importance random forest2](https://github.com/user-attachments/assets/5094b9ec-e086-4ec2-b34a-fc5c375428c5)


