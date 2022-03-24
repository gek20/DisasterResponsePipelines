import sys
import pandas as pd
import re
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    load the two csv files and join them using 'id' column
    :param messages_filepath:  path to the csv file with the messages
    :param categories_filepath:  path to the csv file with the categories
    :return: dataframe with all the info
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.concat([messages, categories], axis=1, join="inner")
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def extract_categories_info(df):
    '''
    extract the categories from the dataset
    :param df: daaframe
    :return: dataframe with the categories
    '''
    categories = df['categories'].str.split(";", expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: re.split('^(.*?)-', x)[1])
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]

        # convert column from string to numeric
        categories[column] = categories[column].astype("int")
    df = df.drop(columns=['categories'])
    df = pd.concat([df, categories], axis=1, join="inner")
    return df


def remove_duplicates(df):
    '''
    check for duplicates and remove them
    :param df: dataframe
    :return: the dataframe without duplicates
    '''
    df.drop_duplicates(subset="id",
                       keep=False, inplace=True)
    duplicated = df[df['id'].isin(df['id'][df['id'].duplicated()])]
    assert len(duplicated['id'].unique()) == 0
    return df


def test_boolean_colums(df):
    '''
    test if there are only 0 and 1 in the dataframe
    :param df: dataframe
    :return: dataframe with the data that are not boolean removed
    '''
    d_targets = df.iloc[:, -36:]
    for c in d_targets.columns:
        df.drop(df[(df[c] != 0) & (df[c] != 1)].index, inplace=True)
    return df


def remove_colums_with_zero_variance(df):
    '''
    if there is 0 variance in a columns it means that it is not an important feature
    :param df: dataframe
    :return: dataframe with the coloumn with 0 variance removed
    '''
    rows = df.shape[1]
    df = df.loc[:, (df != df.iloc[0]).any()]
    print("Removed {} column/columns with 0 variance".format(str(rows - df.shape[1])))
    return df


def clean_data(df):
    '''
    step by step cleaning of the dataframe
    :param df: dataframe
    :return: cleaned dataframe
    '''
    df = extract_categories_info(df)
    df = remove_duplicates(df)
    df = test_boolean_colums(df)
    # df = remove_colums_with_zero_variance(df)
    return df


def save_data(df, database_filename):
    '''
    save data in database
    :param df: dataframe
    :param database_filename: name of the db
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('Messages', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
