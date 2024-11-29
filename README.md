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
- Identified an optimal max depth of 18, which improved the model's performance for both models (All Attributes and Selected Attributes) 
![Accuracy](https://github.com/user-attachments/assets/d830bdef-9a02-4f27-bbda-49d286d1576c)

<img src="https://github.com/user-attachments/assets/a1a24890-3b13-463b-81db-1d2a1abdd62a" width="600" />

<img src="https://github.com/user-attachments/assets/c78e54cc-f789-47bb-a035-cb6d2cf3c2d0" width="600" />

<img src="https://github.com/user-attachments/assets/6a9dd71a-a8e9-492b-87ef-da59e1112b87" width="600" />


## Feature Engineering:
### > FILE: 
    - Feature Engineering.sas
#### Highlights:
- Binned numeric attributes using domain-specific logic, enhancing the performance of both the all-attributes and selected-attributes models.
- Documented the binning logic for reproducibility.

