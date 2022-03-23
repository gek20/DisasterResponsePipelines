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


### Data Analysis:

1. ETL :
    - Removed rows with unexpected values (not 0 or 1)
    - Removed duplicated rows
    - Extracted labels information

### Data Visualization:
One of the main problem of the dataset is that the distribution of the data is really unbalanced. 
Each message could have multiple labels but some of them are really rare.  

![VIsulaization of the unbalanced classes](https://github.com/gek20/DisasterResponsePipelines/blob/6b578e0b151ee532be6425cd2843c73fd2e63be5/pictures/Class%20Distributions.png)

The label "child_alone" is never present, this means that is not giving any useful information at the model during the training.

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

![Correlation Between Labels](https://github.com/gek20/DisasterResponsePipelines/blob/937e32c0f68412c4eb9081c4295ff772f90a569c/pictures/correlation_matrix_labels.png)
6. Acknowledgements
 
Udacity for providing the material for the Data Science Nanodegree Program.
Dataset provided by ![Appen](https://appen.com/) (Figure Eight).