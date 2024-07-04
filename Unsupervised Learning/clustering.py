#!/usr/bin/env python
# Dependencies

import numpy as np
import sklearn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from yellowbrick.cluster.elbow import KElbowVisualizer
from yellowbrick.datasets.loaders import load_nfl
from sklearn.cluster import KMeans
import yellowbrick

# -----------
def clustering(n_samples, n_c,n_f):
    # ------- Create a dataset 
    X, y = make_blobs(n_samples = n_samples, n_features = n_f, 
                      centers = n_c, cluster_std = 0.5, 
                      random_state = 42)
    
    # ------- Creating Air Polution dataset 
    # Consider y labeles to be Warning Level (WL), from 0 <= WL <= 4
    df = pd.DataFrame(X, columns=[f'PM{i}'for i in range(X.shape[1])])
    df['Labels'] = y
    df.to_csv('clustering_dataset.csv')

    # ------- Do simple presentation of clusters - pair of most correlated features
    df1 = df[['PM0','PM1','PM2','PM3','PM4']].corr().abs()
    dfn = df1.to_numpy()
    np.fill_diagonal(dfn,0)
    fl = float(np.max(dfn))
    
    # ------- Choose those features to present the result 
    arr = np.where(dfn == fl)[0]
    cols = list(df.columns)
    unique_labels = set(y)
    for label in unique_labels:
        plt.scatter(X[y == label][:, arr[0]], X[y == label][:, arr[1]], s=50, label=f'WL_{label}')
    lst = ['PM0','PM1','PM2','PM3','PM4']
    plt.legend(loc='upper right')
    plt.title(f'Cluster presentation of most correlated features in dataset: {cols[arr[0]]} and {cols[arr[1]]}')
    plt.xlabel(lst[arr[0]])
    plt.ylabel(lst[arr[1]])
    plt.savefig('initial_cluster')
    plt.show()
    
    # ------- Find best K for K-means 
    elbowfinder = KElbowVisualizer(KMeans(), k=10, locate_elbow=True)
    elbowfinder.fit(X)
    plt.show()
    num_clusters = int(input('Enter number of clusters from visualization: '))
    plt.savefig('elbomwthod')
    
    # ------- Perform the K-means
    kmeans = KMeans(n_clusters = num_clusters, init = 'k-means++', 
                    max_iter = 300, n_init = 10, random_state=42)
    kmeans.fit(df)
    df['Predicted labels'] = kmeans.predict(df)
    predicted_labels = df['Predicted labels']
    
    
    # ------- Visualize the results
    # Getting the Centroids
    centroids = kmeans.cluster_centers_
    label = kmeans.fit_predict(df)
    unique_labels = np.unique(label)
    for label in unique_labels:
        plt.scatter(X[y == label][:, arr[0]], X[y == label][:, arr[1]], s=50, label=f'WL_{label}')
    plt.scatter(centroids[:,arr[0]], centroids[:,arr[1]], s = 80, color = 'k')
    plt.legend()
    plt.savefig('final_clusters')
    plt.show()



'''--------------'''
def main():
    clustering(2000,5,5)

'''--------------'''
if __name__ == "__main__":
    main()
