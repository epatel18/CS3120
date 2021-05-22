import os
import numpy as np
import matplotlib.pyplot as plt
import sklearn.svm as svm
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def getData():
    csvPath = "/MNIST/train.csv"
    path = os.getcwd() + csvPath
    data = np.array(pd.read_csv(path))
    return (
            data[0:5000,1:], 
            data[0:5000,0], 
            data[5000:7900,1:], 
            data[5000:7900,0], 
            data[7900:8900,1:], 
            data[7900:8900,0])
    
def modelController(xTrain, yTrain, xTest, yTest, xVal, yVal):
    models = {1: svmModel, 2: linRegModel, 3: decTreeModel}
    for i in models:
        models[i](xTrain, yTrain, xTest, yTest, xVal, yVal)
        
def svmModel(xTrain, yTrain, xTest, yTest, xVal, yVal):
    Gamma = .001
    C = 1
    model = svm.SVC(kernel='poly', C=C, gamma=Gamma)
    report(xTrain, yTrain, xTest, yTest, model, xVal, yVal)
    print("SVM Model Predictions\n")
    
def linRegModel(xTrain, yTrain, xTest, yTest, xVal, yVal):
    model = LogisticRegression(max_iter=10000)
    report(xTrain, yTrain, xTest, yTest, model, xVal, yVal)
    print("Logistic Regression Predictions\n")
    
def decTreeModel(xTrain, yTrain, xTest, yTest, xVal, yVal):
    model = DecisionTreeClassifier()
    report(xTrain, yTrain, xTest, yTest, model, xVal, yVal)
    print("Decision Tree Classifier Predictions\n")

def report(xTrain, yTrain, xTest, yTest, model, xVal, yVal):
    model.fit(xVal, yVal)
    yPred=model.predict(xTest)
    print(classification_report(yTest, yPred))
    Showlist=np.arange(10)
    for i in Showlist:
        sample = xTest[i]
        sample = sample.reshape((28,28))
        plt.imshow(sample,cmap='gray')
        plt.title('The prediction is: ' + str(yPred[i]))
        plt.show()

if __name__ == '__main__':
    xTrain, yTrain, xTest, yTest, xVal, yVal = getData()
    modelController(xTrain, yTrain, xTest, yTest, xVal, yVal)