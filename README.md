# Branch 3

## Selected Attributes Model Development:
### > FILE: 
    - Selected Attributes Model Development.sas
#### Highlights:
- Created a decision tree model with selected attributes, achieving superior performance compared to the all-attributes model
  - Used a max depth of 15, to ensure consistency with All Attributes Model 
![SELECTED training](https://github.com/user-attachments/assets/98841e98-8500-44d3-9345-d172d5df9c0e)
![SELECTED TEST](https://github.com/user-attachments/assets/d7f5aac9-4028-447c-b0c7-bceb3f46bd67)

+ Baseline - All Attributes Model results are seen below for comparison (*as seen in Branch 1*)
<img src="https://github.com/user-attachments/assets/1cb235d0-05cc-4a02-8c4f-dbac11f279f4" width="650" />
<img src="https://github.com/user-attachments/assets/2817706e-1a48-4cc9-b25c-25ee6ee95f4f" width="650" />


## Hyperparameter Tuning:
### > FILE: 
    - Hyperparameter Tuning.sas
#### Highlights:
- Experimented with different max depth values for the All Attributes and Selected Attributes Decision Tree models
- Identified an optimal max depth of 18, which improved the model's performance for both models (All Attributes and Selected Attributes) 
![md15vsmd18](https://github.com/user-attachments/assets/dc3e2945-afd7-4c61-9d45-23cb08e0a16b)


## Feature Engineering:
### > FILE: 
    - Feature Engineering.sas
#### Highlights:
- Binned numeric attributes using domain-specific logic, enhancing the performance of both the all-attributes and selected-attributes models
- Documented the binning logic for reproducibility
![accuracy](https://github.com/user-attachments/assets/c0f6d7b8-ea5f-4051-ab6e-af35d608c569)

<img src="https://github.com/user-attachments/assets/a48d4dc5-5270-4eda-ad84-68ff14e27204" width="600" />

<img src="https://github.com/user-attachments/assets/56781eb2-88a2-4ae7-bc97-efc069c8ae49" width="600" />

<img src="https://github.com/user-attachments/assets/b7ebdd8f-c599-4b94-8784-4a681c667669" width="600" />




