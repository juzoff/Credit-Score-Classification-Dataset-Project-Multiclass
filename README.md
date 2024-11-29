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
![Accuracy](https://github.com/user-attachments/assets/7ef3fd75-e744-469c-a9a1-0ae227d007ee)

<img src="https://github.com/user-attachments/assets/0c87a54e-8ad4-4f00-82d8-b98e8a3e364b" width="600" />

<img src="https://github.com/user-attachments/assets/b0f65e0a-a029-40bf-a52a-dca8edf1aed1" width="600" />

<img src="https://github.com/user-attachments/assets/e8fb65fe-3862-452a-8d1a-9865ab9d791a" width="600" />


## Feature Engineering:
### > FILE: 
    - Feature Engineering.sas
#### Highlights:
- Binned numeric attributes using domain-specific logic, enhancing the performance of both the all-attributes and selected-attributes models.
- Documented the binning logic for reproducibility.

