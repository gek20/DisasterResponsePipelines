# Disaster Response Pipeline Project

### Instructions:
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