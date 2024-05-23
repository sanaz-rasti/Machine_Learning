#!/usr/bin/env python

# Dependencies
import numpy as np
import pandas as pd
import umap
import plotly.express as px
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA, FastICA, FactorAnalysis
from sklearn.manifold import TSNE, Isomap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# -------------------------------------
def DimensionalityReduction():
    # Load Dataset
    datasetdf = pd.read_csv(input('Enter Dataset path: '))
            
    df1 = datasetdf._get_numeric_data()
    df1 = df1.dropna()
    ind2spxy = int((len(df1.index)/4)*3)
    
    x_train = df1[:ind2spxy].to_numpy()
    x_train = normalize(x_train)
    x_test  = df1[ind2spxy:].to_numpy()
    ncomp = int(input(f"""Data has {x_train.shape[1]} features, how many features would you like to include?"""))
    
    model_name = input("""Choose: 
                        Component/Factor Based model:
                            a) PCA: Principal Component Analysis
                            b) ICA: Independant Component Analysis
                            c) Factor Analysis
                        Projection Based Model:
                            d) t-SNE
                            e) ISOMAP
                            f) UMAP (metric='cosine')
                            """)

    if model_name =='a':
        model = PCA(n_components = ncomp)
    elif model_name =='b':
        model = FastICA(n_components=ncomp)
    elif model_name =='c':
        model = FactorAnalysis(n_components=ncomp,random_state=0)
    elif model_name =='d':
        x_train = x_train[:1000,:]
        print(f'For the TSNE model, we reduce the sample size to 1000, x_train shape={x_train.shape}')
        model = TSNE(n_components=ncomp,method='exact',learning_rate='auto',init='random', perplexity=3)
    elif model_name =='e':
        x_train = x_train[:1000,:]
        print(f'For the ISOMAP model, we reduce the sample size to 1000, x_train shape={x_train.shape}')
        model = Isomap(n_components=ncomp)
    elif model_name =='f':
        x_train = x_train[:1000,:]
        print(f'For the UMAP model, we reduce the sample size to 1000, x_train shape={x_train.shape}')
        model = umap.UMAP(n_components=ncomp, metric='cosine')
        
    x_reduced = model.fit_transform(x_train)

    plt.scatter(x_train[:,0], x_train[:,1],c=x_train[:,2],cmap='cool')
    plt.title("Scatter plot of original data")
    plt.show()

    plt.scatter(x_reduced[:,0], x_reduced[:,1],c=x_reduced[:,2],cmap='cool')
    plt.title("Scatter plot of dimensionality reduced data")
    plt.show()

    return x_reduced



'''--------------------------------------------------------------------------------------'''
'''--------------------------------------------------------------------------------------'''
'''--------------------------------------------------------------------------------------'''
def main():
    x_reduced = DimensionalityReduction()

''' ------------------------------------------------------------------------'''
if __name__ == "__main__":
    main()










