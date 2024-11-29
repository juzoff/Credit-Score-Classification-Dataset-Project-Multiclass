# Branch 3

## Selected Attributes Model Development:
### > FILE: 
    - Selected Attributes Model Development.sas
#### Highlights:
- Created a decision tree model with selected attributes, achieving superior performance compared to the all-attributes model
  - Used a max depth of 15, to ensure consistency with All Attributes Model 
![SELECTED training](https://github.com/user-attachments/assets/98841e98-8500-44d3-9345-d172d5df9c0e)
![SELECTED TEST](https://github.com/user-attachments/assets/d7f5aac9-4028-447c-b0c7-bceb3f46bd67)

+ All Attributes Model results are seen below for comparison (*as seen in Branch 1*)
<img src="https://github.com/user-attachments/assets/c820c532-4bc3-4853-95a1-65fcc841a697" width="650" />
<img src="https://github.com/user-attachments/assets/5000743b-99dd-462b-851b-039563c5a6d6" width="650" />

## Hyperparameter Tuning:
### > FILE: 
    - Hyperparameter Tuning.sas
#### Highlights:
- Experimented with different max depth values for the All Attributes and Selected Attributes Decision Tree models
- Identified an optimal max depth of 18, which improved the model's performance for both model 


## *>Branch 3<*
### Selected Attributes Model Deployment:
- Created a decision tree model with selected attributes (*see Branch 2 for removed attributes in this model*), achieving superior performance compared to the all-attributes model.
### Hyperparameter Tuning:
- Experimented with different max depth values for decision trees.
- Identified an optimal depth of 18, which improved the model's performance.
### Feature Engineering:
- Binned numeric attributes using domain-specific logic, enhancing the performance of both the all-attributes and selected-attributes models.
- Documented the binning logic for reproducibility.

