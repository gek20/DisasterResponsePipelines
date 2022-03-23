import pickle
import sys
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB


def load_data(database_filepath):
    '''
    load the data from the database
    :param database_filepath: path to the database
    :return: pandas dataframes for model input e labels and list of labels
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Messages', con=engine)
    X = df['message'].values
    Y = df.iloc[:, -36:]
    # print(Y.sum())
    y = Y.values
    target_names = list(Y.columns)
    return X, y, target_names


def tokenize(text):
    '''
    function to tokenize the sentence removing stop_words
    :param text: the input sentence
    :return: the tokens list
    '''
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return tokens


def build_model():
    '''
    create the classification model
    :return: model pipeline
    '''

    pipeline = Pipeline([
        ('text_pipeline', Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize,
                                     ngram_range=(1, 2),
                                     max_df=0.75)),
            ('tfidf', TfidfTransformer())
        ])),
        ('clf', MultiOutputClassifier(estimator=RandomForestClassifier()))
    ])
    pipeline = test_model(pipeline)
    return pipeline


def evaluate_model(model, x_test, y_test, category_names):
    '''
    evaluate the model and print result
    :param model: model to evaluate
    :param X_test: test samples
    :param Y_test: test labels
    :param category_names: name of the classes
    '''
    y_pred = model.predict(x_test)
    cr = classification_report(y_test, y_pred, target_names=category_names)
    print(cr)


def save_model(model, model_filepath):
    '''
    save the model in a pickle format
    :param model: model
    :param model_filepath: path where we want to store the model
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def test_model(model):
    '''
    testing different parameters of the model using GridSearch and returning the best one
    :param model: the pipeline we want to test
    :return: gridsearch model. -> the fit() function will return the model with the best parameters
    '''
    # print(model.get_params())
    parameters = {
        'clf__estimator': [RandomForestClassifier(), KNeighborsClassifier()],
        'text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'text_pipeline__vect__max_df': (0.75, 1.0),
        'text_pipeline__vect__max_features': (None, 2000, 5000)
    }
    cv = GridSearchCV(model, param_grid=parameters, verbose=10)

    return cv


def create_correlation_matrix(Y, class_names):
    '''
    check if there is a correlation among classes and save the graph
    :param classes: the lables for the model
    '''
    df = pd.DataFrame(Y, columns=class_names)
    corr_matrix = df.corr()
    fig, ax = plt.subplots(figsize=(25, 15))
    ax.set_title('Correlation Matrix classes', fontsize=16)
    sn.heatmap(corr_matrix, annot=True, cmap='Blues', fmt='.1g')
    plt.savefig('./pictures/correlation_matrix_labels.png', dpi=300)


def main():
    '''
    main function of the code, load the data, create and train a classification model and save it as pickle
    '''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)

        create_correlation_matrix(Y, category_names)  # create and saving correlation matrix for the labels

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=20)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)
        print("best score: {:.3f}, best params: {}".format(model.best_score_, model.best_params_))

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
