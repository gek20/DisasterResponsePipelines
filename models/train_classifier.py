import pickle
import sys
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
import pandas as pd

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
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE


def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Messages', con=engine)
    X = df['message'].values
    Y = df.iloc[:, -35:]
    y = Y.values
    target_names = list(Y.columns)
    return X, y, target_names


def tokenize(text):
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return tokens


def build_model():
    # pipeline = Pipeline([
    #     ('text_pipeline', Pipeline([
    #         ('vect', CountVectorizer(tokenizer=tokenize)),
    #         ('tfidf', TfidfTransformer())
    #     ])),
    #     #('sampling', SMOTE()),
    #     ('clf', MultiOutputClassifier(RandomForestClassifier(class_weight='balanced')))
    # ])

    from imblearn.pipeline import Pipeline

    text_pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize,
                                                       ngram_range=(1, 2),
                                                       max_df=0.75)),
                              ('tfidf', TfidfTransformer(sublinear_tf=True))])
    pipeline = Pipeline([('features', FeatureUnion([('text_pipeline', text_pipeline)])),
                         ('clf', MultiOutputClassifier(RandomForestClassifier(class_weight='balanced')))])


    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    cr = classification_report(Y_test, y_pred, target_names=category_names)
    print(cr)


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def test_model(model, X_train, y_train):
    parameters = {
        'clf__estimator': [KNeighborsClassifier(), RandomForestClassifier(class_weight='balanced'), RidgeClassifier(class_weight='balanced')]}
    # MLPClassifier(hidden_layer_sizes=(128),activation='relu',solver='adam',batch_size=500,shuffle=True)
    cv = GridSearchCV(model, param_grid=parameters, scoring='accuracy',verbose=10)

    cv.fit(X_train, y_train)

    print("best score: {:.3f}, best params: {}".format(cv.best_score_, cv.best_params_))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Testing parameters...')
        test_model(model, X_train, Y_train)
        # best score: 0.246, best params: {'clf__estimator': RandomForestClassifier(class_weight='balanced')}

        print('Training model...')
        model.fit(X_train, Y_train)

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
