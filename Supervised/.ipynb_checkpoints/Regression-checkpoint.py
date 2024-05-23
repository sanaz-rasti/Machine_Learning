from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.datasets import load_iris, load_diabetes, load_digits, load_linnerud, load_wine
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as pyplot
from sklearn.datasets import make_classification
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn import metrics

class regression:
    # Class Variables
    y_pred = []
    regscore = []
    
    # ---- Dataset ----
    dataset = input("""Enter Dataset: 
                    a) Iris
                    b) Digits
                    c) Linnerud
                    d) Wine""")
    
    if dataset == 'a':
        Data = load_iris()
        print('Dataset: Iris')
    elif dataset == 'b':
        Data = load_digits()
        print('Dataset: Digits')
    elif dataset == 'c':
        Data = load_linnerud()
        print('Dataset: Linnerud')
    elif dataset == 'd':
        Data = load_wine()
        print('Dataset: Wine')

    X = Data.data
    y = Data.target

    # Split train and test data 
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size = 0.2,
                                                        random_state = 42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initiation 
    def __init__(self,*args):
        scaler = StandardScaler()
        self.X_train = regression.X_train 
        self.X_test  = regression.X_test
        self.y_train = regression.y_train 
        self.y_test  = regression.y_test

    
    def fitmodel(self):
        
        inpt = input('''Choose an option: 
                a) Linear Regression
                b) Decision Tree Regressor
                c) Support Vector Regressor
                d) Losso Regression 
                e) Random Forest Regressor''')
              
        if inpt == 'a':
            print("Model: Linear Regression")
            # Linear Regression
            model = LinearRegression()          
            
        elif inpt == 'b':
            print("Model: Decision Tree Regression")
            model = DecisionTreeRegressor(max_depth=2)
            
        elif inpt == 'c':
            print("Model: Support Vector Regression")
            model = SVR()
            
        elif inpt == 'd':
            print("Model: Losso Regression")
            model = linear_model.Lasso(alpha=0.1)

        elif inpt == 'e':
            print("Random Forest Regressor")
            model = RandomForestRegressor(max_depth=3)

        # --- Fit the model
        model.fit(self.X_train, self.y_train)

        # ----------------------------
        # Determination of prediction: 
        # Regression Score: (1 - u/v): u = ((y_true - y_pred)**2).sum(),
        # v = ((y_true - y_true.mean())**2).sum()
        self.__class__.regscore = model.score(self.X_train, self.y_train)

        # --- Predictions
        y_pred = model.predict(self.X_test)
        self.__class__.y_pred = y_pred

        errors = list()
        for i in range(len(self.y_test)):
            err = (self.y_test[i] - y_pred[i])**2
            errors.append(err)
            # print('>%.1f, %.1f = %.3f' % (self.y_test[i], y_pred[i], err))

        
        # plot errors
        pyplot.plot(errors)
        pyplot.xticks(ticks=[i for i in range(len(errors))])
        pyplot.xlabel('Data Points')
        pyplot.ylabel('Mean Squared Error')
        pyplot.show()


''' ------------------------------------------------------------------------'''
def main():
    regressor = regression() 
    regressor.fitmodel()

''' ------------------------------------------------------------------------'''
if __name__ == "__main__":
    main()
