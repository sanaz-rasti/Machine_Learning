#!/usr/bin/env python
from prompter import yesno
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

class Forecast:
    def __init__(self,*args):
        datapath = input('Datapath: ')
        self.df = pd.read_csv(datapath)
        
    def forcast_model(self):
        self.df['FLOODS'].replace(['YES', 'NO'], [1,0], inplace=True)
        X = self.df.drop(["FLOODS","REGION"],axis= 1)
        Y = self.df["FLOODS"]
        if len(self.df) <= 10**5:
            less_features = yesno('Logically few features are important:')
            if less_features:
                best_features = SelectKBest(score_func=chi2, k=3)
                features = best_features.fit(X,Y)
                df_scores  = pd.DataFrame(features .scores_)
                df_columns = pd.DataFrame(X.columns)
                features_scores = pd.concat([df_columns, df_scores], axis=1)
                features_scores.columns = ['Features', 'Score']
                features_scores = features_scores.sort_values(by = 'Score', ascending=False)
                X = self.df[list(features_scores.Features[:4])]
                Y = self.df[['FLOODS']]
                X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.4,random_state=100)
                # model, lasso:
                # The Lasso optimizes a least-square problem with a L1 penalty. 
                # --> However impossible to optimize a logistic function with the Lasso
                # --> Employ Logistic Regression Estimator with L1 penalty
                est = LogisticRegression(penalty='l1', solver='liblinear')
                est.fit(X_train,y_train)

            else: 
                X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.4,random_state=100)
                # model: Support Vector Classifier 
                est = SVC(kernel = 'linear')
                est.fit(X_train,y_train)
                
        else:
            # print('Try SGD Regressor')
            X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.4,random_state=100)
            # model: Stochastic Gradient Decent Classifier 
            est = SGDClassifier(loss = 'log_loss', penalty='elasticnet')
            est.fit(X_train,y_train)
            
        # predict
        y_pred = est.predict(X_test)
        
        # evaluation 
        print(f'Accuracy:{metrics.accuracy_score(y_test, y_pred)}')
        print(f'Recall: {metrics.recall_score(y_test, y_pred, zero_division=1)}')
        print(f'Precision:{metrics.precision_score(y_test, y_pred, zero_division=1)}')
        print(f'Classification Report:{metrics.classification_report(y_test, y_pred, zero_division=1)}')
        try:    
            y_pred_proba = est.predict_proba(X_test)[::,1]            
            false_positive_rate, true_positive_rate, _ = metrics.roc_curve(y_test, y_pred_proba) 
            auc= metrics.roc_auc_score(y_test, y_pred_proba)
            plt.plot(false_positive_rate, true_positive_rate,label="AUC=" + str(auc))
            plt.title('ROC Curve')
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.legend(loc=4)
        except:
            pass
            
        
'''--------------------------------------------------------------------------------------'''
def main():
    forecaster = Forecast() 
    forecaster.forcast_model()

''' ------------------------------------------------------------------------'''
if __name__ == "__main__":
    main()  