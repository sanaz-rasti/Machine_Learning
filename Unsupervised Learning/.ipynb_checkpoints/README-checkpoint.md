# Description
This folder contains four usupervised Machine Learning approaches of Associate Rule Mining, Clustering, Dimensionality Reduction and Feature Extraction. Ther algorithms are implemented and tested using publicly available datasets. The details of data can be found in section 1 and the Algorithms are listed and described in section 2.  

## 1. Datasets
### 1.1 'retaildata.csv'
link: https://www.kaggle.com/datasets/shedai/retail-data-set
<br>n_samples: 33356
<br>n_features: 8

### 1.2 'GroceryStoreDataSet'
link: https://www.kaggle.com/datasets/shazadudwadia/supermarket
<br>n_samples: 19
<br>n_features: 1

### 1.3 'Dwellings_totalNZ-wide_format_updated_16-7-20.csv'
link: https://www.stats.govt.nz/information-releases/statistical-area-1-dataset-for-2018-census-updated-march-2020/
<br>n_samples: 32521 
<br>n_features: 176


## 2. Algorithms

### 2.1 Associate Rule Mining:

It is a rule-based machine learning algorithm for uncovering the hidden relationship between variables. It is a popular data mining tool for analysis of social network, market basket, retail data, fraud deetction, etc..

**Module**: 'association_rule_mining.ipynb'
<br>The notebook implemented using retail data and grocery stode data. 
The first part of analysis investigate sum of discount for each customer on each specific date. 
In the second part, the 'arm' function is written to suggest the frequent products in the grocery dataset employing Associate Rule Mining. For each antecedent product a conseqent product is suggested by 'arm'. Moreover for each item, three metric for associate rule evaluation are reported. These metrics are:
* Support: Transactions(itemX,itemY) / totalTransactions
* Confidence: Transactions(itemX,itemY) / Transactions(itemX)
* Lift: Transactions(itemX,itemY) * Transactions(itemX) / Transactions(itemY)


### 2.2 Clustering:
**Module**: 





### 2.3 Dimensionality Reduction:
The advantage of transforming data from a high-dimensional into a low-dimensional space is inevitable while dealing with sparse, computationaly interactable datasets. It simplififies the analysis and visualisation of underlying patterns in dataset.  
There are two main dimensionality reduction method of Component/Factor based and Projection based; examples of each method and their usecases are presented herein:

**A) Component/Factor based**  
**A.1) PCA**: Numerous applications can benefit from PCA, including : Neuroscience, Quantitative finance, Market research and indexes of attitude, Population genetics, Development indexes, Residential differentiation, Human Intelligence. PCA is performing well on certain data, if the variance of data reserved with fair number of selected principal components.

**A.2) ICA**: Data with several Guassian feature distributions, and/or statistical independant feature sets are bet fitted for ICA dimensionality reduction. It has widely applied for neuron image data analysis, stuck market price, finance data, astronomy data, color image data, etc..

**A.3) Factor Analysis**:

**B) Project based**
**B.1) t-SNE**
**B.2) ISOMAP**:
**B.3) UMAP**: 
    
**Module**: 'dimensionality_reduction.py'
<br>After loading the dataset, the user will be informed the number of features in the dataset. Then input the desired number of features to be returned by the application.   
There are six-algorithms for feature dimensionality reduction in this module: 
1. PCA: Principal Component Analysis
2. ICA: Independant Component Analysis
3. Factor Analysis
4. t-SNE: t-distributed Stochastic Neighbor Embedding
5. ISOMAP: ISOmetric MAPing
6. UMAP(metric='cosine'): Uniform Manifold Approximation and Projection

The **DimensionalityReduction()** function returns the data in lower dimension (x_reduced). 


### 2.4 Feature Extraction:
**Module**: 'feature_extraction.py'
<br>
**1) Low Variance Filter:**
**2) Univariate feature Selection:**
**3) High Correlation Filter:**
**4) Random Forest Regressor:**
**5) Recursive Feature Elimination:**
**6) Forward Feature Selection:**



















