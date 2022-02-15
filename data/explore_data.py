import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Messages', con=engine)
    data = df.iloc[:, -35:]
    return data


def find_cluster(data):
    sil = []
    for i in range(2, 22):
        kmeans = KMeans(init='k-means++', n_clusters=i, n_init=100)
        kmeans.fit(data)
        clusters_clients = kmeans.predict(data)
        silhouette_avg = silhouette_score(data, clusters_clients)
        sil.append(silhouette_avg)
        print('score de silhouette: {:<.3f}'.format(silhouette_avg), str(i))
    return sil


def plot_sil(scores):
    import matplotlib.pyplot as plt
    y = np.arange(start=2, stop=22, step=1)
    plt.plot(y, scores, color='#e17055')
    plt.title('Siluhette values for "No Personal Info"')
    plt.xlabel('# cluster')
    plt.ylabel('cost')
    plt.savefig('tutti_no_pi.png', dpi=300)


def main():
    if len(sys.argv) == 2:

        database_filepath = sys.argv[1]
        data=load_data(database_filepath)
        sil_score= find_cluster(data)
        plot_sil(sil_score)
        #print(data.sum())

if __name__ == '__main__':
    main()