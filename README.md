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


For example: (report)

    Was the phrase "women and children first" just a myth or did they really try to get out women and children first? If so, they would show as more likely to survive.

    are women more likely to survive? plot barplot x-axis is sex and y-axis is survival rate
    are children more likely to survive? bin age into 0-16, 17+, plot barplot on x-axis where y is survival rate (new variable = is_child)
    run chi-square test sex + survival
    run a chi-square test is_child + survival
    run a t-test on age and survived

Statistical tests (report.ipynb)

    At least 2 statistical tests are included in your final report.

    The correct tests are run, given the data type and distribution, and the correct conclusions are drawn. For example (other tests may be used):

        correlation: 2 continuous variables, normally distributed, testing for LINEAR correlation only (H_0: Not linearly dependent)

        independent t-test: 1 continuous, somewhat normally distributed variable, one boolean variable, equal variance, independent (H_0: population mean of each group is equal)

        chi-square test: 2 discrete variables. (H_0: the 2 variables are independent of each other).

Summary (report.ipynb)

    Following your exploration section, you summarize your analysis (in a markdown cell using natural language): what you found and how you will use it moving forward.

    This includes key takeaways from all the questions answered in explore, a list of which features will be used in modeling and why, and which features will not move forward and why. You may only call out a few of these features in the presentation, but having that there for reference is important in a report. A group of features may have the same reason why, and those can be mentioned together.

Modeling
Select Evaluation Metric (Report.ipynb)



    Clear communication as to how you evaluated and compared models.

    What metric(s) did you use and why? For example, in one case, you may decide to use precision over accuracy. If so, why? If you use multiple metrics, how will you decide which to select if metric is better for model A but another is better for model B? Will you rank them? Find a way to aggregate them into a single metric you can use to rank?

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


Evaluate Top Model on Test (Report.ipynb)

    Your top performing model, and only your top performing model should be evaluated on your test dataset. The purpose of having a test dataset to evaluate only the final model on is to have an estimate of how the model will perform in the future on data it has never seen.





























