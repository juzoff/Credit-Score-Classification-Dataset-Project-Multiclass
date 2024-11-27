/* INSERTING BALANCED CSV DATASET (WHICH WAS CREATED IN RSTUDIO)*/
/* 1. Read the file in SAS and display the contents using the PROC IMPORT and PROC PRINT procedures */
proc import /* out keyword is used to name a table */
	out=balanced_data_creditscore

	/* Datafile keyword takes the path of the file from the hard disk */
	datafile="/home/u63872294/Data/balanced_data_creditscore.csv"

	/* “dbms= csv replace” is telling SAS it is a csv file. */
	dbms=csv replace;
	/* “Getnames=yes” will use the first line of the csv file as column names */
	getnames=yes;

	/* Data keyword takes the name of the SAS table imported as balanced_data_creditscore */
run;

/* Print the first 100 observations of the dataset */
proc print data=balanced_data_creditscore (obs=100);
run;

/* Using contents procedure to check metadata */
proc contents data=balanced_data_creditscore;
run;

/*Double Checking Balanced Class Distribution*/
/* Display class distribution of the Credit_Score variable */
proc freq data=balanced_data_creditscore;
	tables Credit_Score;
run;










/* ALL ATTRIBUTES - 15 max depth */
/* Step 1: Load the balanced dataset */
proc import datafile="/home/u63872294/Data/balanced_data_creditscore.csv" 
		out=balanced_data_creditscore dbms=csv replace;
	getnames=yes;
run;

/* Step 2: Check the class distribution */
proc freq data=balanced_data_creditscore;
	tables Credit_Score / out=ClassDist;
run;

/* Step 3: Split the balanced dataset into training (70%) and testing (30%) */
proc surveyselect data=balanced_data_creditscore out=TrainData samprate=0.7 
		seed=12345 /* Set seed for reproducibility */
		outall;
	strata Credit_Score;
run;

data TrainData TestData;
	set TrainData;

	if selected then
		output TrainData;
	else
		output TestData;
run;

/* Step 4: Create a decision tree model with updated maxdepth for multi-class classification */
proc hpsplit data=TrainData maxdepth=15 seed=12345;
	/* Set seed for reproducibility */
	class Credit_Score Credit_Mix Occupation Last_Loan_1 Last_Loan_2 Last_Loan_3 
		Last_Loan_4 Last_Loan_5 Last_Loan_6 Last_Loan_7 Last_Loan_8 Last_Loan_9 
		Payment_Behaviour Payment_of_Min_Amount;
	model Credit_Score=Age Amount_invested_monthly Annual_Income 
		Changed_Credit_Limit Credit_History_Age Credit_Mix Credit_Utilization_Ratio 
		Delay_from_due_date Interest_Rate Last_Loan_1 Last_Loan_2 Last_Loan_3 
		Last_Loan_4 Last_Loan_5 Last_Loan_6 Last_Loan_7 Last_Loan_8 Last_Loan_9 
		Monthly_Balance Monthly_Inhand_Salary Num_Bank_Accounts Num_Credit_Card 
		Num_Credit_Inquiries Num_of_Delayed_Payment Num_of_Loan Occupation 
		Outstanding_Debt Payment_Behaviour Payment_of_Min_Amount Total_EMI_per_month;
	grow gini;

	/* Use the Gini index for classification */
	prune costcomplexity;

	/* Prune the tree */
	code file='/home/u63872294/Data/creditscore_decision_tree_score.sas';

	/* Generate scoring code */
run;

/* Step 5: Score the test dataset */
data ScoredTestData;
	set TestData;
	%include '/home/u63872294/Data/creditscore_decision_tree_score.sas';

	/* Apply scoring code */
	/* Correct scoring logic for multi-class classification */
	if P_Credit_ScoreGood >=max(P_Credit_ScorePoor, P_Credit_ScoreStanda) then
		Predicted_Credit_Score='Good';
	else if P_Credit_ScorePoor >=max(P_Credit_ScoreGood, P_Credit_ScoreStanda) then
		Predicted_Credit_Score='Poor';
	else if P_Credit_ScoreStanda >=max(P_Credit_ScoreGood, P_Credit_ScorePoor) then
		Predicted_Credit_Score='Standa';
run;

/* Step 5.1: Score the training dataset */
data ScoredTrainData;
	set TrainData;
	%include '/home/u63872294/Data/creditscore_decision_tree_score.sas';

	/* Apply scoring code */
	/* Correct scoring logic for multi-class classification */
	if P_Credit_ScoreGood >=max(P_Credit_ScorePoor, P_Credit_ScoreStanda) then
		Predicted_Credit_Score='Good';
	else if P_Credit_ScorePoor >=max(P_Credit_ScoreGood, P_Credit_ScoreStanda) then
		Predicted_Credit_Score='Poor';
	else if P_Credit_ScoreStanda >=max(P_Credit_ScoreGood, P_Credit_ScorePoor) then
		Predicted_Credit_Score='Standa';
run;

/*---*/
/* View structure of the scored test dataset */
proc contents data=ScoredTestData;
run;

/*---*/
/* Step 6: Evaluate the performance for the test set */
proc freq data=ScoredTestData;
	tables Credit_Score*Predicted_Credit_Score / norow nocol nopercent chisq;
run;

/* Step 6.1: Evaluate the performance for the training set */
proc freq data=ScoredTrainData;
	tables Credit_Score*Predicted_Credit_Score / norow nocol nopercent chisq;
run;







/* RANDOM FOREST - ALL ATTRIBUTES - sorted by Gini*/
/* Step 1: Load the balanced dataset */
proc import datafile="/home/u63872294/Data/balanced_data_creditscore.csv" 
    out=balanced_data_creditscore 
    dbms=csv 
    replace;
    getnames=yes;
run;

/* Step 2: Check the class distribution */
proc freq data=balanced_data_creditscore;
    tables Credit_Score / out=ClassDist;
run;

/* Step 3: Split the balanced dataset into training (70%) and testing (30%) */
proc surveyselect data=balanced_data_creditscore 
    out=TrainData 
    samprate=0.7 
    seed=12345 
    outall;
    strata Credit_Score;
run;

data TrainData TestData;
    set TrainData;
    if selected then output TrainData;
    else output TestData;
run;

/* Step 4: Create a Random Forest model for multi-class classification */
proc hpforest data=TrainData maxtrees=100 seed=12345;
    /* Set seed for reproducibility */
    target Credit_Score / level=nominal;
    /* Specify target variable and its level */
    
    /* Specify numeric input variables */
    input Age Amount_invested_monthly Annual_Income Changed_Credit_Limit Credit_History_Age
          Credit_Utilization_Ratio Delay_from_due_date Interest_Rate Monthly_Balance
          Monthly_Inhand_Salary Num_Bank_Accounts Num_Credit_Card Num_Credit_Inquiries
          Num_of_Delayed_Payment Num_of_Loan Outstanding_Debt Total_EMI_per_month / level=interval;
    
    /* Specify categorical input variables */
    input Credit_Mix Occupation Last_Loan_1 Last_Loan_2 Last_Loan_3 Last_Loan_4 Last_Loan_5 
          Last_Loan_6 Last_Loan_7 Last_Loan_8 Last_Loan_9 Payment_Behaviour 
          Payment_of_Min_Amount / level=nominal;
    
    /* Save the Random Forest model */
    save file="/home/u63872294/Data/hpforest_model";
    
    /* Output variable importance table */
    ods output VariableImportance=VarImp;
run;

/* Step 5: Sort the Variable Importance table by Gini (Loss Reduction) */
proc sort data=VarImp;
    by descending Gini;  /* Sort by Gini, if Gini is the variable for loss reduction importance */
run;

/* Step 5.1: Display the sorted results */
proc print data=VarImp;
    var Variable Gini;  /* Adjust 'Variable' and 'Gini' based on actual column names */
run;

/* Step 6: Score the test dataset using Random Forest */
data ScoredTestData;
    set TestData;
    /* Apply scoring logic based on Random Forest model */
    /* Include the model file */
    %include "/home/u63872294/Data/hpforest_model.sas"; /* Corrected the file extension */
    
    /* Random Forest produces the probabilities for each class */
    /* Choose the class with the highest probability */
    if P_Credit_ScoreGood >= max(P_Credit_ScorePoor, P_Credit_ScoreStanda) then
        Predicted_Credit_Score = 'Good';
    else if P_Credit_ScorePoor >= max(P_Credit_ScoreGood, P_Credit_ScoreStanda) then
        Predicted_Credit_Score = 'Poor';
    else if P_Credit_ScoreStanda >= max(P_Credit_ScoreGood, P_Credit_ScorePoor) then
        Predicted_Credit_Score = 'Standa';
run;

/* Step 5.2: Score the training dataset using Random Forest */
data ScoredTrainData;
    set TrainData;
    /* Apply scoring logic based on Random Forest model */
    /* Include the model file */
    %include "/home/u63872294/Data/hpforest_model.sas"; /* Corrected the file extension */
    
    /* Random Forest produces the probabilities for each class */
    /* Choose the class with the highest probability */
    if P_Credit_ScoreGood >= max(P_Credit_ScorePoor, P_Credit_ScoreStanda) then
        Predicted_Credit_Score = 'Good';
    else if P_Credit_ScorePoor >= max(P_Credit_ScoreGood, P_Credit_ScoreStanda) then
        Predicted_Credit_Score = 'Poor';
    else if P_Credit_ScoreStanda >= max(P_Credit_ScoreGood, P_Credit_ScorePoor) then
        Predicted_Credit_Score = 'Standa';
run;

/* Step 6.3: Evaluate the performance for the test set */
proc freq data=ScoredTestData;
    tables Credit_Score*Predicted_Credit_Score / norow nocol nopercent chisq;
run;

/* Step 6.4: Evaluate the performance for the training set */
proc freq data=ScoredTrainData;
    tables Credit_Score*Predicted_Credit_Score / norow nocol nopercent chisq;
run;

