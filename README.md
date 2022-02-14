# Disaster Response Pipeline Project


### Project Components

0. Data 
Disaster data from Appen 

2. ETL Pipeline
In a Python script, process_data.py, write a data cleaning pipeline that:

Loads the messages and categories datasets
Merges the two datasets
Cleans the data
Stores it in a SQLite database
2. ML Pipeline
In a Python script, train_classifier.py, write a machine learning pipeline that:

Loads data from the SQLite database
Splits the dataset into training and test sets
Builds a text processing and machine learning pipeline
Trains and tunes a model using GridSearchCV
Outputs results on the test set
Exports the final model as a pickle file
3. Flask Web App
We are providing much of the flask web app for you, but feel free to add extra features depending on your knowledge of flask, html, css and javascript. For this part, you'll need to:

Modify file paths for database and model as needed
Add data visualizations using Plotly in the web app. One example is provided for you


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

1. Data Cleaning:
    - Removed rows with unexpected values (not 0 or 1)
    - Removed column with 0 variance ()

![VIsulaization of the unbalanced classes](https://github.com/gek20/DisasterResponsePipelines/blob/6b578e0b151ee532be6425cd2843c73fd2e63be5/pictures/Class%20Distributions.png)