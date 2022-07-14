# Telco_Churn_Analysis
Codeup Class Project Kalpana


### Project Planning

Mon - Explore
Tues - MVP
Wed - Further Exploration and updates
    - Practice delivery
Thurs - Present



### Data Dictionary

All column labels in original data set are self-explanatory

Target Variable: Churn


### Steps to reproduce

You will need an env.py file that contains the hostname, username and password of the mySQL database that contains the telco_churn db. Store that env file locally in the repository.

Clone my repo (including the acquire.py and prepare.py) (confirm .gitignore is hiding your env.py file)

Libraries used are pandas, matplotlib, seaborn, numpy, sklearn, and scipy.stats.


### Project Goals

The goal is to determine the main areas of churn and enable the investors to make data-based decisions on where to invest resources to maintain or grow your customer base. 


### Project Description

Companies depend on a stong customer base. Determining why people are leaving the company can help make informed decisons about how to modify company policy to the benefit of the customers, and therefore shareholders. 


### Initial Testing and Hypotheses

    Does internet service cause churn? If yes, is it specific to a type of internet?

    Does payemnet type cause churn?

    What are the major causes of churn?

    Are the major causes of churn entire groups? or subgroups? ie. all payment methods are equally terrible, or is it just one payment type?

### Report findings






### Detailed Project Plan

Acquire

Requires:

    env.user
    env.password
    env.host

acquire.get_telco_data()

    Function gets telco_data first from save file inside folder, then, if no file exits, it will pull directly from mysql.

acquire.new_telco_data()

    pulls data from mysql

Cleaning

    dropped all total_charges values that were ' ' (belonged to brand new customers who had no total payment values as they were too new)
    converted total_charges to 'float' type

Prep Telco

    drop all 'id' columns
    created dummy columns for:
        contract
        internet
        payment
    encoded binary catagory for:
        gender
        partner
        dependents
        phone services
        multiple_lines
        online_security
        online_backup
        device_protection
        tech_support
        streaming_tv
        streaming_movies
        paperless builing
        churn

Split Telco

train, validate, test = prepare.split_telco_data()

    20% of data into test group
    30% of remaining data into validate group (30% of 80% = 24% of total data)
    70% of remaining data into train group (70% of 80% = 56% of total data)

target leakage

    data is further split to avoid target leakage

    x_train = train.drop(columns=['churn'])
    y_train = train.churn

    x_validate = validate.drop(columns=['churn'])
    y_validate = validate.churn

    x_test = test.drop(columns=['churn'])
    y_test = test.churn



Explore

    Does internet service cause churn? If yes, is it specific to a type of internet?

    Does payemnet type cause churn?

    What are the major causes of churn?

    Are the major causes of churn entire groups? or subgroups? ie. all payment methods are equally terrible, or is it just one payment type?


For example:

Visual:
    graph all object variables coding for churn (bar graph, histograms)
    graph continuous variables coding for churn (scatterplot)
    
Statistical: 
    chi^2 statistics for all object variables

Statistical tests (report.ipynb)

        chi-square test: 2 discrete variables. (H_0: the 2 variables are independent of each other)
        create loop for testing for object variables

Summary

Features based on visual analysis:

catagories where 40% or more of that group churned:

    Electronic checking
    Fiber Optic
    Senior
    Month-to-month
    No Tech support
    Online-backup
    Device-Protection  
    
    
    
    
 Features based on statistical analysis:

      ['partner',
     'dependents',
     'multiple_lines',
     'online_security',
     'online_backup',
     'device_protection',
     'tech_support',
     'streaming_tv',
     'streaming_movies',
     'paperless_billing',
     'contract_type',
     'internet_service_type',
     'payment_type']  

    





Modeling
Select Evaluation Metric

 precision is highest in m1
 recall is highest in m3

If giving promotion deals to convice people to stay:
    
    If you want to ensure you get all people likely to churn and are willing to risk a few extra people getting the promotion ----> maximize recall
    
    If you only want people likely to churn to get the promotion, and are willing to miss a few members of the target audience to ensure no loyal customers get the promotion ----> maximze precison    
    
 I suggest maximizing recall, getting a wider audience with a few extra people getting a promotion    
    
    

Evaluate Baseline

    The baseline is based on the mode of the target value. The mode is that the customer did not churn. The baseline results in a test that's 73% accurate.

Develop 3 Models:
Evaluate on Train
Evaluate on Validate 

    The 3 models can differ based on the features used, the hyperparameters selected, and/or the algorithm used to fit the data.
    

    Model 1: 
        RandomForest
        Visulization Exploration Subgroup
        Baseline: 73.4%
        max depth 5
        train: 78.5% 
        validate: 76.9%

    Model 2:
        RandomForest
        Visulization Exploration Group
        Baseline: 73.4%
        max depth 4
        train: 78.1% 
        validate: 77.1%

    Model 3: 
        RandomForest
        Statistical Exploration Group
        Baseline: 73.4%
        max depth 5
        train: 78.5% 
        validate: 78.3%


Evaluate Top Model on Test

    Your top performing model, and only your top performing model should be evaluated on your test dataset. The purpose of having a test dataset to evaluate only the final model on is to have an estimate of how the model will perform in the future on data it has never seen.





























