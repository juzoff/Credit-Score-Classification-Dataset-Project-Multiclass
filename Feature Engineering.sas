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









/*ALL ATTRIBUTES - Binned - 18 max depth*/
/* Step 1: Load the balanced dataset */
proc import datafile="/home/u63872294/Data/balanced_data_creditscore.csv" 
    out=balanced_data_creditscore dbms=csv replace;
    getnames=yes;
run;

/* Step 2: Check the class distribution */
proc freq data=balanced_data_creditscore;
    tables Credit_Score / out=ClassDist;
run;

/* Step 2.1: Create bins for attributes */
data balanced_data_creditscore_binned;
    set balanced_data_creditscore;

    /* Manual binning for Age */
    if Age < 25 then Age_Binned = '18-24';
    else if Age < 41 then Age_Binned = '25-40';
    else if Age < 61 then Age_Binned = '41-60';
    else Age_Binned = '61+';

    /* Refined binning for Credit_History_Age (Months) */
    if Credit_History_Age <= 12 then Credit_History_Age_Binned = 'Very Short';
    else if Credit_History_Age <= 60 then Credit_History_Age_Binned = 'Short';
    else if Credit_History_Age <= 120 then Credit_History_Age_Binned = 'Moderate';
    else if Credit_History_Age <= 240 then Credit_History_Age_Binned = 'Long';
    else Credit_History_Age_Binned = 'Very Long';

    /* Refined binning for Annual_Income (USD) */
    if Annual_Income < 30000 then Annual_Income_Binned = 'Low';
    else if Annual_Income < 70000 then Annual_Income_Binned = 'Medium';
    else if Annual_Income < 120000 then Annual_Income_Binned = 'High';
    else Annual_Income_Binned = 'Very High';

    /* Binning for Delay_from_due_date */
    if Delay_from_due_date < 0 then Delay_Binned = 'Early Payment';
    else if Delay_from_due_date <= 7 then Delay_Binned = 'Slight Delay';
    else if Delay_from_due_date <= 30 then Delay_Binned = 'Moderate Delay';
    else if Delay_from_due_date <= 60 then Delay_Binned = 'Severe Delay';
    else Delay_Binned = 'Critical Delay';

    /* Binning for Interest_Rate */
    if Interest_Rate <= 5 then Interest_Rate_Binned = 'Low';
    else if Interest_Rate <= 15 then Interest_Rate_Binned = 'Moderate';
    else Interest_Rate_Binned = 'High';

    /* Binning for Num_Bank_Accounts */
    if Num_Bank_Accounts <= 2 then Bank_Accounts_Binned = 'Few';
    else if Num_Bank_Accounts <= 5 then Bank_Accounts_Binned = 'Moderate';
    else Bank_Accounts_Binned = 'Many';

    /* Binning for Num_Credit_Card */
    if Num_Credit_Card <= 2 then Credit_Cards_Binned = 'Few';
    else if Num_Credit_Card <= 5 then Credit_Cards_Binned = 'Moderate';
    else Credit_Cards_Binned = 'Many';

	/* Binning for Num_Credit_Inquiries */
	if Num_Credit_Inquiries = 0 then Num_Credit_Inquiries_Binned = 'None';
	else if Num_Credit_Inquiries <= 3 then Num_Credit_Inquiries_Binned = 'Low';
	else if Num_Credit_Inquiries <= 8 then Num_Credit_Inquiries_Binned = 'Moderate';
	else if Num_Credit_Inquiries <= 12 then Num_Credit_Inquiries_Binned = 'High';
	else Num_Credit_Inquiries_Binned = 'Very High';

    /* Binning for Num_of_Delayed_Payment */
    if Num_of_Delayed_Payment = 0 then Delayed_Payment_Binned = 'None';
    else if Num_of_Delayed_Payment <= 3 then Delayed_Payment_Binned = 'Low';
    else if Num_of_Delayed_Payment <= 6 then Delayed_Payment_Binned = 'Moderate';
    else Delayed_Payment_Binned = 'High';

    /* Binning for Outstanding_Debt */
    if Outstanding_Debt < 1000 then Debt_Binned = 'Low';
    else if Outstanding_Debt < 3000 then Debt_Binned = 'Moderate';
    else Debt_Binned = 'High';

    /* Binning for Changed_Credit_Limit */
    if Changed_Credit_Limit < 0 then Credit_Limit_Change_Binned = 'Decrease';
    else if Changed_Credit_Limit <= 5 then Credit_Limit_Change_Binned = 'Small Increase';
    else if Changed_Credit_Limit <= 20 then Credit_Limit_Change_Binned = 'Moderate Increase';
    else Credit_Limit_Change_Binned = 'Large Increase';

    /* Binning for Monthly_Inhand_Salary */
    if Monthly_Inhand_Salary < 1000 then Inhand_Salary_Binned = 'Low';
    else if Monthly_Inhand_Salary <= 5000 then Inhand_Salary_Binned = 'Medium';
    else Inhand_Salary_Binned = 'High';

	/* Binning for Total_EMI_per_Month */
	if Total_EMI_per_Month <= 100 then EMI_Binned = 'Very Low';
	else if Total_EMI_per_Month <= 200 then EMI_Binned = 'Low';
	else if Total_EMI_per_Month <= 300 then EMI_Binned = 'Moderate';
	else if Total_EMI_per_Month <= 357 then EMI_Binned = 'High';
	else EMI_Binned = 'Very High';

	/* Binning for Amount_Invested_Monthly */
	if Amount_Invested_Monthly <= 500 then Invested_Binned = 'Very Low';
	else if Amount_Invested_Monthly <= 1000 then Invested_Binned = 'Low';
	else if Amount_Invested_Monthly <= 1500 then Invested_Binned = 'Moderate';
	else if Amount_Invested_Monthly <= 1980 then Invested_Binned = 'High';
	else Invested_Binned = 'Very High';

	/* Binning for Monthly_Balance */
	if Monthly_Balance < 100 then Balance_Binned = 'Very Low';
	else if Monthly_Balance <= 500 then Balance_Binned = 'Low';
	else if Monthly_Balance <= 1200 then Balance_Binned = 'Moderate';
	else Balance_Binned = 'High';

	/* Binning for Credit_Utilization_Ratio */
	if Credit_Utilization_Ratio < 20 then Utilization_Binned = 'Very Low';
	else if Credit_Utilization_Ratio <= 30 then Utilization_Binned = 'Low';
	else if Credit_Utilization_Ratio <= 40 then Utilization_Binned = 'Moderate';
	else if Credit_Utilization_Ratio <= 50 then Utilization_Binned = 'High';
	else Utilization_Binned = 'Very High';
run;

/* Step 3: Split the balanced and binned dataset into training (70%) and testing (30%) */
proc surveyselect data=balanced_data_creditscore_binned out=TrainData samprate=0.7 
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

/* Step 4: Create a decision tree model with the binned attributes */
proc hpsplit data=TrainData maxdepth=18 seed=12345;
    class Credit_Score Credit_Mix Occupation Last_Loan_1 Last_Loan_2 Last_Loan_3 
          Last_Loan_4 Last_Loan_5 Last_Loan_6 Last_Loan_7 Last_Loan_8 Last_Loan_9 Payment_of_Min_Amount 
          Age_Binned Credit_History_Age_Binned Annual_Income_Binned 
          Delay_Binned Interest_Rate_Binned Bank_Accounts_Binned 
          Credit_Cards_Binned Delayed_Payment_Binned Debt_Binned 
          Credit_Limit_Change_Binned Inhand_Salary_Binned Num_Credit_Inquiries_Binned Payment_Behaviour
          EMI_Binned Invested_Binned Balance_Binned Utilization_Binned;
    model Credit_Score=Age_Binned Annual_Income_Binned Credit_History_Age_Binned 
          Delay_Binned Interest_Rate_Binned Bank_Accounts_Binned 
          Credit_Cards_Binned Delayed_Payment_Binned Debt_Binned 
          Credit_Limit_Change_Binned Inhand_Salary_Binned Credit_Mix 
          Last_Loan_1 Last_Loan_2 Last_Loan_3 Last_Loan_4 Last_Loan_5 Last_Loan_6 Last_Loan_7 Last_Loan_8 Last_Loan_9
          Occupation Payment_of_Min_Amount Num_Credit_Inquiries_Binned Payment_Behaviour
          EMI_Binned Invested_Binned Balance_Binned Utilization_Binned;
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









/*SELECTED ATTRIBUTES - Binned - 18 max depth*/
/* Step 1: Load the balanced dataset */
proc import datafile="/home/u63872294/Data/balanced_data_creditscore.csv" 
    out=balanced_data_creditscore dbms=csv replace;
    getnames=yes;
run;

/* Step 2: Check the class distribution */
proc freq data=balanced_data_creditscore;
    tables Credit_Score / out=ClassDist;
run;

/* Step 2.1: Create bins for attributes */
data balanced_data_creditscore_binned;
    set balanced_data_creditscore;

    /* Manual binning for Age */
    if Age < 25 then Age_Binned = '18-24';
    else if Age < 41 then Age_Binned = '25-40';
    else if Age < 61 then Age_Binned = '41-60';
    else Age_Binned = '61+';

    /* Refined binning for Credit_History_Age (Months) */
    if Credit_History_Age <= 12 then Credit_History_Age_Binned = 'Very Short';
    else if Credit_History_Age <= 60 then Credit_History_Age_Binned = 'Short';
    else if Credit_History_Age <= 120 then Credit_History_Age_Binned = 'Moderate';
    else if Credit_History_Age <= 240 then Credit_History_Age_Binned = 'Long';
    else Credit_History_Age_Binned = 'Very Long';

    /* Refined binning for Annual_Income (USD) */
    if Annual_Income < 30000 then Annual_Income_Binned = 'Low';
    else if Annual_Income < 70000 then Annual_Income_Binned = 'Medium';
    else if Annual_Income < 120000 then Annual_Income_Binned = 'High';
    else Annual_Income_Binned = 'Very High';

    /* Binning for Delay_from_due_date */
    if Delay_from_due_date < 0 then Delay_Binned = 'Early Payment';
    else if Delay_from_due_date <= 7 then Delay_Binned = 'Slight Delay';
    else if Delay_from_due_date <= 30 then Delay_Binned = 'Moderate Delay';
    else if Delay_from_due_date <= 60 then Delay_Binned = 'Severe Delay';
    else Delay_Binned = 'Critical Delay';

    /* Binning for Interest_Rate */
    if Interest_Rate <= 5 then Interest_Rate_Binned = 'Low';
    else if Interest_Rate <= 15 then Interest_Rate_Binned = 'Moderate';
    else Interest_Rate_Binned = 'High';

    /* Binning for Num_Bank_Accounts */
    if Num_Bank_Accounts <= 2 then Bank_Accounts_Binned = 'Few';
    else if Num_Bank_Accounts <= 5 then Bank_Accounts_Binned = 'Moderate';
    else Bank_Accounts_Binned = 'Many';

    /* Binning for Num_Credit_Card */
    if Num_Credit_Card <= 2 then Credit_Cards_Binned = 'Few';
    else if Num_Credit_Card <= 5 then Credit_Cards_Binned = 'Moderate';
    else Credit_Cards_Binned = 'Many';

	/* Binning for Num_Credit_Inquiries */
	if Num_Credit_Inquiries = 0 then Num_Credit_Inquiries_Binned = 'None';
	else if Num_Credit_Inquiries <= 3 then Num_Credit_Inquiries_Binned = 'Low';
	else if Num_Credit_Inquiries <= 8 then Num_Credit_Inquiries_Binned = 'Moderate';
	else if Num_Credit_Inquiries <= 12 then Num_Credit_Inquiries_Binned = 'High';
	else Num_Credit_Inquiries_Binned = 'Very High';

    /* Binning for Num_of_Delayed_Payment */
    if Num_of_Delayed_Payment = 0 then Delayed_Payment_Binned = 'None';
    else if Num_of_Delayed_Payment <= 3 then Delayed_Payment_Binned = 'Low';
    else if Num_of_Delayed_Payment <= 6 then Delayed_Payment_Binned = 'Moderate';
    else Delayed_Payment_Binned = 'High';

    /* Binning for Outstanding_Debt */
    if Outstanding_Debt < 1000 then Debt_Binned = 'Low';
    else if Outstanding_Debt < 3000 then Debt_Binned = 'Moderate';
    else Debt_Binned = 'High';

    /* Binning for Changed_Credit_Limit */
    if Changed_Credit_Limit < 0 then Credit_Limit_Change_Binned = 'Decrease';
    else if Changed_Credit_Limit <= 5 then Credit_Limit_Change_Binned = 'Small Increase';
    else if Changed_Credit_Limit <= 20 then Credit_Limit_Change_Binned = 'Moderate Increase';
    else Credit_Limit_Change_Binned = 'Large Increase';

    /* Binning for Monthly_Inhand_Salary */
    if Monthly_Inhand_Salary < 1000 then Inhand_Salary_Binned = 'Low';
    else if Monthly_Inhand_Salary <= 5000 then Inhand_Salary_Binned = 'Medium';
    else Inhand_Salary_Binned = 'High';
run;

/* Step 3: Split the balanced and binned dataset into training (70%) and testing (30%) */
proc surveyselect data=balanced_data_creditscore_binned out=TrainData samprate=0.7 
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

/* Step 4: Create a decision tree model with the binned attributes */
proc hpsplit data=TrainData maxdepth=18 seed=12345;
    class Credit_Score Credit_Mix Occupation Last_Loan_1 Last_Loan_2 Last_Loan_3 
          Last_Loan_4 Last_Loan_5 Payment_of_Min_Amount 
          Age_Binned Credit_History_Age_Binned Annual_Income_Binned 
          Delay_Binned Interest_Rate_Binned Bank_Accounts_Binned 
          Credit_Cards_Binned Delayed_Payment_Binned Debt_Binned 
          Credit_Limit_Change_Binned Inhand_Salary_Binned Num_Credit_Inquiries_Binned;
    model Credit_Score=Age_Binned Annual_Income_Binned Credit_History_Age_Binned 
          Delay_Binned Interest_Rate_Binned Bank_Accounts_Binned 
          Credit_Cards_Binned Delayed_Payment_Binned Debt_Binned 
          Credit_Limit_Change_Binned Inhand_Salary_Binned Credit_Mix 
          Last_Loan_1 Last_Loan_2 Last_Loan_3 Last_Loan_4 Last_Loan_5
          Occupation Payment_of_Min_Amount Num_Credit_Inquiries_Binned;
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
