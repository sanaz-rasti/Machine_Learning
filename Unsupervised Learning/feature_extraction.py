#!/usr/bin/env python

# Dependencies
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import normalize
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_classif,RFE,SequentialFeatureSelector
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# ------------------------------
def FeatureExtranction():
    # Load Dataset
    datasetdf = pd.read_csv(input('Enter Dataset path: '))
            
    df1 = datasetdf._get_numeric_data()
    df1 = df1.dropna()
    ind2spxy = int((len(df1.index)/4)*3)
    
    x_train = df1[:ind2spxy].to_numpy()
    x_test  = df1[ind2spxy:].to_numpy()
    ncomp = int(input(f"""Data has {x_train.shape[1]} features, how many features would you like to include?"""))
    
    model_name = input("""Choose model:
                            a) Low Variance Filter
                            b) Univariate feature Selection
                            c) High Correlation Filter 
                            d) Random Forest Regressor
                            e) Recursive Feature Elimination
                            f) Forward Feature Selection
                            """)
    #---------------
    if model_name =='a':
        df1 = pd.DataFrame(normalize(df1), columns=df1.columns)
        variances = df1.var()
        varmean = np.mean(variances)
        df = variances.to_frame()
        for k in list(df.keys()):
            df.drop(df[df[k] <= varmean].index, inplace = True)
        df.rename(columns={0: "Variance"}, inplace=True)
        print('Features with variance higher than Mean-Var:\n' )
        print(f'Features: {df}')
        
        plt.plot(variances)
        plt.axhline(varmean, color='r', linestyle='dashed', label = 'Mean Variance')
        plt.xticks(rotation = 90)
        plt.title('Data Variance')
        plt.legend()
        plt.show()
        print('The model returns the variance dataframe. \n ----- \n')
        return df
    #---------------
    elif model_name =='b':
        # Univariate approaches examine each feature on its own. It examines its features, ranks them, 
        # and selects the features accordingly on a set of criteria. The disadvantage is that the 
        # link between the attributes is ignored.
        
        df1 = pd.DataFrame(normalize(df1), columns=df1.columns)
        cols = list(df1.columns)
        X_train = df1[cols[:-1]]
        Y_train = df1[cols[-1]]
        print(f'X TRAIN shape: {X_train.shape}')
        
        selector = SelectKBest(f_classif, k=ncomp)
        x_new = selector.fit_transform(X_train, Y_train)
        print(f'X NEW Shape: {x_new.shape}')
        new_feature_set = selector.get_feature_names_out()
        print(f'\n {ncomp} Selected Features:\n {new_feature_set}')
        print('--------\nThe model returns the new feature set. \n')
        return new_feature_set
        
    #---------------    
    elif model_name =='c':
        df1 = pd.DataFrame(normalize(df1), columns=df1.columns)
        s= df1.corr().stack(-1)
        s = s[s != 1]
        corr = s.unstack()
        dfcorr = corr.abs().unstack().sort_values(ascending=False).to_frame().dropna()
        dfcorr.rename(columns={0: "Correlation"}, inplace=True)
        print(f'Descending Sorted Correlaion between features of dataset:\n-------\nFeatures:{dfcorr}')
        print('-------')
        new_feature_set = list(set([item for t in list(dfcorr[:10].index) for item in t]))
        print(f'\nGetting feature names for the 10 highest correlations:\n \n{new_feature_set}')
        print('-----\nThe model returns the new feature set. \n')
        return new_feature_set
    #---------------    
    elif model_name =='d':
        df = pd.read_csv('retaildata.csv')
        print("""The Random Forest Regressor is put into practice for the retail data. 
        Considering the Price and Discount of an Item as X-Data and the bought-Quantities of the Item as the Y-Data. 
        After training, the regressor is expected to predict the quantities based on given pair of Price and Discount for an Item. \n""")

        df1 = df[['Price','Discount','Quantity']].dropna()
        df1 = pd.DataFrame(normalize(df1), columns=df1.columns)
        model = RandomForestRegressor(random_state=1, max_depth=10)
        
        # Split train and test 
        X_train = df1[['Price','Discount']][:20000]
        Y_train = df1[['Quantity']][:20000]
        
        X_test = df1[['Price','Discount']][20000:]
        Y_test = df1[['Quantity']][20000:]
        
        # train the model, to get an estimator for a given pair of 'Price' and 'Discount', how many items will be boughtin the store? 
        model.fit(X_train,Y_train)
        ypred = list(model.predict(X_test))
        ytrue = list(Y_test['Quantity'])
        fig = px.scatter(x=ytrue, 
                         y=ypred, 
                         labels={'x': 'Quantity ground truth', 'y': 'Quantity prediction'}, 
                         title="""Prediction of X_test data, scatter matrix of (Y-True,Y-Preds)""")

        fig.add_shape(type="line", line=dict(dash='dash'),x0=min(ytrue), y0=min(ytrue),x1=max(ytrue), y1=max(ytrue))
        fig.show()
        # model.score: coefficient of determination
        # Negative score means prediction of test data using the trained model is not 
        # following trend of data to match the target values
        sc = model.score(X_test, Y_test)
        print(f'Coeff of Determin SC = {sc}')
        print('------\nThe model returns the fitted Random Forest Regressor model. \n')
        return model
    #--------------- 
    elif model_name =='e':
        df1 = pd.DataFrame(normalize(df1), columns=df1.columns)
        cols = list(df1.columns)
        X_train = df1[cols[:-1]]
        Y_train = df1[cols[-1]]
        print(f'X TRAIN shape: {X_train.shape}')
        estimator = SVR(kernel="linear")
        selector = RFE(estimator, n_features_to_select=ncomp, step=1)
        x_new = selector.fit_transform(X_train, Y_train)
        print(f'X NEW Shape: {x_new.shape}')
        new_feature_set = selector.get_feature_names_out()
        print(f'\n {ncomp} Selected Features:\n {new_feature_set}')
        print('-----\nThe model returns the new feature set. \n')
        return new_feature_set
    
    #---------------
    elif model_name =='f':
        df1 = pd.DataFrame(normalize(df1), columns=df1.columns)
        cols = list(df1.columns)
        X_train = df1[cols[:-1]]
        Y_train = df1[cols[-1]]
        print(f'X TRAIN shape: {X_train.shape}')
        estimator = SVR(kernel="linear")
        selector = SequentialFeatureSelector(estimator, n_features_to_select=ncomp)
        selector.fit(X_train, Y_train)
        x_new = selector.transform(X_train)
        print(f'X NEW Shape: {x_new.shape}')
        new_feature_set = selector.get_feature_names_out() 
        print(f'\n {ncomp} Selected Features:\n {new_feature_set}')
        print('------\n The model returns the new feature set.\n')
        return new_feature_set




'''--------------------------------------------------------------------------------------'''
'''--------------------------------------------------------------------------------------'''
'''--------------------------------------------------------------------------------------'''
def main():
    FeatureExtranction()

''' ------------------------------------------------------------------------'''
if __name__ == "__main__":
    main() 





