from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_diabetes, load_digits, load_linnerud, load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import metrics

# --------
class classifier:
    
    # Class Variables
    y_pred = []
    confusion_matrix = []
    
    # ---- Dataset ----
    dataset = input("""Enter Dataset: 
                    a) Iris
                    b) Diabetes
                    c) Digits
                    d) Linnerud
                    e) Wine""")
    
    if dataset == 'a':
        Data = load_iris()
        print('Dataset: Iris')
    elif dataset == 'b':
        Data = load_diabetes()
        print('Dataset: Diabetes')
    elif dataset == 'c':
        Data = load_linnerud()
        print('Dataset: Digits')
    elif dataset == 'd':
        Data = load_wine()
        print('Dataset: Linnerud')
    elif dataset == 'e':
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
        self.X_train = classifier.X_train 
        self.X_test  = classifier.X_test
        self.y_train = classifier.y_train 
        self.y_test  = classifier.y_test

    
    def clsf(self):
        
        inpt = input('''Choose an option: 
                a) SVM
                b) KNN
                c) Random Forest Classifier
                d) Logistic Regression ''')
              
        if inpt == 'a':
            print("Model: SVM")
            model = make_pipeline(StandardScaler(), SVC(gamma='auto'))
            
        elif inpt == 'b':
            print("Model: KNN")
            model = KNeighborsClassifier(n_neighbors = 3)
            
        elif inpt == 'c':
            print("Model: Random Forest Classifier")
            model = RandomForestClassifier(max_depth=2, random_state=0)
            
        elif inpt == 'd':
            print("Model: Logistic Regression")
            model = LogisticRegression()

        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        
        # Computing the confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        self.__class__.y_pred = y_pred
        self.__class__.confusion_matrix = cm  
        
        # ----------- Evaluating the model
        accuracy = metrics.accuracy_score(self.y_test, self.__class__.y_pred)
        precision = metrics.precision_score(self.y_test, self.__class__.y_pred, average='macro')
        recall = metrics.recall_score(self.y_test, self.__class__.y_pred, average='macro')

        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
    
    
    def plot_cm(self):
        
        cm = self.__class__.confusion_matrix
        
        # Plotting the confusion matrix using Matplotlib
        fig, ax = plt.subplots(figsize = (8, 8))
        im = ax.imshow(cm, interpolation = 'nearest', cmap = plt.cm.Blues)
        ax.figure.colorbar(im, ax = ax)
        ax.set(xticks = np.arange(cm.shape[1]),
               yticks = np.arange(cm.shape[0]),
               xticklabels = classifier.Data.target_names, yticklabels = classifier.Data.target_names,
               title  = 'Confusion Matrix',
               ylabel = 'True label',
               xlabel = 'Predicted label')
        plt.setp(ax.get_xticklabels(), 
                 rotation=45, ha="right", 
                 rotation_mode="anchor")
        fmt = 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha = "center", va = "center",
                        color = "white" if cm[i, j] > thresh else "black")
        fig.tight_layout()

'''--------------------------------------------------------------------------------------'''
'''--------------------------------------------------------------------------------------'''
'''--------------------------------------------------------------------------------------'''
def main():
    clsfire = classifier() 
    clsfire.clsf()
    clsfire.plot_cm()

''' ------------------------------------------------------------------------'''
if __name__ == "__main__":
    main() 
