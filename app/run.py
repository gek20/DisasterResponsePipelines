import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine

app = Flask(__name__, template_folder='template')


def tokenize(text):
    '''
    tokenize the input sentence to use it as input for our model
    :param text: the input sentence
    :return: list of tokens
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data (only one dot is needed on Windows)
engine = create_engine('sqlite:///./data/DisasterResponse.db')
df = pd.read_sql_table('Messages', engine)

# load model
model = joblib.load("./models/classifier.pkl")


def create_distribution_graph(x, y, title, x_label, y_label):
    graph = {
        'data': [
            Bar(
                x=x,
                y=y
            )
        ],

        'layout': {
            'title': title,
            'yaxis': {
                'title': x_label
            },
            'xaxis': {
                'title': y_label
            }
        }
    }

    return graph


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    graphs = []
    # extract data needed for visuals

    # graph 1
    # group by type of message histogram
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    graphs.append(
        create_distribution_graph(genre_names, genre_counts, 'Distribution of Message Genres', "Count", "Genre"))

    # graph 2
    # group by categories to find out the most frequent combinations
    d_cat = df.iloc[:, -36:]
    most_frequent_comb = d_cat.groupby(d_cat.columns.tolist()).size().reset_index(). \
        rename(columns={0: 'records'}).sort_values('records', ascending=False)
    most_frequent_comb['cat'] = most_frequent_comb.apply(lambda x: x.index[x == 1].tolist(), axis=1)
    most_frequent_comb['cat'] = most_frequent_comb['cat'].apply(lambda x: " <br> ".join(x))  # add new line in labels
    most_frequent_comb = most_frequent_comb.set_index('cat')
    graphs.append(
        create_distribution_graph(list(most_frequent_comb.head(10).index), most_frequent_comb.head(10)['records'],
                                  'Most Frequent Combinations (TOP 10)', "Count", ""))

    # graph 3
    most_freq_cat=d_cat.sum().sort_values(ascending=False)
    graphs.append(
        create_distribution_graph(list(most_freq_cat.head(10).index), most_freq_cat.head(10),
                                  'Most Frequent Categories (TOP 10)', "Count", "Categories"))

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()
