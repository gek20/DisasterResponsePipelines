# Disaster Response Pipeline Project


### Project Components

0. Data:  
The Disaster data are collected and made available for us from Appen. The data are contained in 2 csv files ("disaster_categories.csv" and "disaster_messages.csv") in the data directory of the project.  

1. ETL Pipeline:  
The data are loaded from the csv files, merged, processed and cleaned. Then data are stored in a SQL db.

2. Data Pipelines:  
Then the data are used to train a pipeline that classify the input messages in multiple classes. The code will use GridSearch to find the best parameters of the pipeline. The trained model is saved as pkl in the models directory of this project.

3. Flask App:  
An interface allows the users to input a message and visualize the classification of the message using a Flask Web App



### Instructions on how to use the code: 
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage

![App screenshot](https://github.com/gek20/DisasterResponsePipelines/blob/d2bb1ceb68e9c9051ce68a52462f6fb632dfba2e/pictures/Screenshot_working_app.PNG)


### Data Analysis:

1. ETL :
    - Removed rows with unexpected values (not 0 or 1)
    - Removed duplicated rows
    - Extracted labels information

### Data Visualization:
One of the main problem of the dataset is that the distribution of the data is really unbalanced. 
Each message could have multiple labels but some of them are really rare.  

![VIsulaization of the unbalanced classes](https://github.com/gek20/DisasterResponsePipelines/blob/6b578e0b151ee532be6425cd2843c73fd2e63be5/pictures/Class%20Distributions.png)

The label "child_alone" is never present, this means that is not giving any useful information to the model during the training.

| Labels                 | Count |
|------------------------|------:|
| related                | 20003 |
| request                |  4447 |
| offer                  |   117 |
| aid_related            | 10805 |
| medical_help           |  2075 |
| medical_products       |  1309 |
| search_and_rescue      |   723 |
| security               |   471 |
| military               |   857 |
| child_alone            |     0 |
| water                  |  1664 |
| food                   |  2904 |
| shelter                |  2299 |
| clothing               |   402 |
| money                  |   602 |
| missing_people         |   297 |
| refugees               |   872 |
| death                  |  1188 |
| other_aid              |  3432 |
| infrastructure_related |  1701 |
| transport              |  1194 |
| buildings              |  1325 |
| electricity            |   530 |
| tools                  |   159 |
| hospitals              |   283 |
| shops                  |   120 |
| aid_centers            |   309 |
| other_infrastructure   |  1147 |
| weather_related        |  7265 |
| floods                 |  2139 |
| storm                  |  2432 |
| fire                   |   282 |
| earthquake             |  2449 |
| cold                   |   527 |
| other_weather          |  1373 |
| direct_report          |  5049 |

The most frequents combinations in our dataset are presented in the following pictures. We can notice how the most frequent combinations is without category because the dataset is full of messages that are not related an emergency. 

![Top 10 Combinations](https://github.com/gek20/DisasterResponsePipelines/blob/d2bb1ceb68e9c9051ce68a52462f6fb632dfba2e/pictures/TOP_10_combinations.png)

We can notice also from the following correlation matrix how some categories are frequently correlated:


![Correlation Between Labels](https://github.com/gek20/DisasterResponsePipelines/blob/937e32c0f68412c4eb9081c4295ff772f90a569c/pictures/correlation_matrix_labels.png)


### Quality of the Model:

From the classification report on our model we can notice how the model works really bad for some categories. This is due to the fact that the dataset is really unbalanced and that the model had not enough data to learn how to predict that categories. A possible solution could be to use some balancing teqniques to improve the quality of the model (SMOTE, oversampling, undersampling,...).

| class                  | precision | recall | f1-score | support |
|------------------------|-----------|--------|----------|---------|
| related                | 0.84      | 0.94   | 0.88     | 4009    |
| request                | 0.82      | 0.49   | 0.61     | 927     |
| offer                  | 0.00      | 0.00   | 0.00     | 20      |
| aid_related            | 0.76      | 0.68   | 0.72     | 2194    |
| medical_help           | 0.63      | 0.11   | 0.19     | 427     |
| medical_products       | 0.88      | 0.22   | 0.35     | 258     |
| search_and_rescue      | 0.72      | 0.08   | 0.15     | 160     |
| security               | 0.25      | 0.01   | 0.02     | 99      |
| military               | 0.65      | 0.10   | 0.17     | 157     |
| child_alone            | 0.00      | 0.00   | 0.00     | 0       |
| water                  | 0.80      | 0.50   | 0.62     | 353     |
| food                   | 0.80      | 0.73   | 0.76     | 585     |
| shelter                | 0.81      | 0.55   | 0.65     | 461     |
| clothing               | 0.76      | 0.25   | 0.38     | 87      |
| money                  | 1.00      | 0.03   | 0.06     | 127     |
| missing_people         | 0.00      | 0.00   | 0.00     | 66      |
| refugees               | 0.54      | 0.09   | 0.15     | 166     |
| death                  | 0.82      | 0.36   | 0.50     | 247     |
| other_aid              | 0.57      | 0.07   | 0.12     | 713     |
| infrastructure_related | 0.00      | 0.00   | 0.00     | 344     |
| transport              | 0.80      | 0.13   | 0.22     | 256     |
| buildings              | 0.78      | 0.27   | 0.40     | 269     |
| electricity            | 0.60      | 0.05   | 0.10     | 112     |
| tools                  | 0.00      | 0.00   | 0.00     | 36      |
| hospitals              | 0.00      | 0.00   | 0.00     | 65      |
| shops                  | 0.00      | 0.00   | 0.00     | 28      |
| aid_centers            | 0.00      | 0.00   | 0.00     | 60      |
| other_infrastructure   | 0.00      | 0.00   | 0.00     | 234     |
| weather_related        | 0.86      | 0.72   | 0.78     | 1484    |
| floods                 | 0.93      | 0.51   | 0.66     | 450     |
| storm                  | 0.78      | 0.65   | 0.71     | 501     |
| fire                   | 1.00      | 0.02   | 0.03     | 61      |
| earthquake             | 0.89      | 0.77   | 0.83     | 483     |
| cold                   | 0.56      | 0.14   | 0.23     | 104     |
| other_weather          | 0.62      | 0.09   | 0.15     | 287     |
| direct_report          | 0.76      | 0.36   | 0.49     | 1044    |
|                        |           |        |          |         |
| micro avg              | 0.81      | 0.56   | 0.66     | 16874   |
| macro avg              | 0.56      | 0.25   | 0.30     | 16874   |
| weighted avg           | 0.75      | 0.56   | 0.60     | 16874   |
| samples avg            | 0.66      | 0.50   | 0.52     | 16874   |

6. Acknowledgements
 
Udacity for providing the material for the Data Science Nanodegree Program.
Dataset provided by [Appen](https://appen.com/) (Figure Eight).