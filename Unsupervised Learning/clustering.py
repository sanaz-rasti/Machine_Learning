#!/usr/bin/env python

# Dependencies
from sklearn.cluster import KMeans, MeanShift
from sklearn.datasets import make_blobs 
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import numpy as np
import random

# -----------
def clustering(cntrs):
    # Dataset
    X, y = make_blobs(n_samples=20000, 
                  centers=cntrs, 
                  n_features=100,
                  random_state=0)
    plt.scatter(X[:,1:3],X[:,4:6],c = X[:,4:6] ,cmap='cool')
    plt.title('Sample data scatter plot')
    plt.show()
    
    model_name = input("""Choose model: 
                        a) K-Means
                        b) Mean Shift\n""")
    
    if model_name == 'a':
        model = KMeans(n_clusters=cntrs, random_state=0, n_init="auto")
            
    elif model_name == 'b':
        model = MeanShift(bandwidth=cntrs)

    
    clus  = model.fit(X)
    labels = clus.labels_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    cluster_centers = clus.cluster_centers_

    plt.figure(1)
    plt.clf()
    
    N = min(n_clusters_,10)
    colors = []
    for i in range(N):
        r = lambda: random.randint(120,240)
        color = '#{:02x}{:02x}{:02x}'.format(r(), r(), r())
        colors.append(color)
    
    markers = ["x"]*N
    for k, col in zip(range(n_clusters_), colors):
        my_members = labels == k
        cluster_center = cluster_centers[k]
        plt.plot(X[my_members, 0], X[my_members, 1], markers[k], color=col)
        plt.plot(
            cluster_center[0],
            cluster_center[1],
            markers[k],
            markerfacecolor=col,
            markeredgecolor="k",
            markersize=14)
        
    plt.title("Estimated number of clusters: %d, showing only %d clusters " % (n_clusters_, N))
    plt.show()



# -----------
def main():
    clustering(10)

# -----------
if __name__ == "__main__":
    main() 

