import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_auc_score,roc_curve, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sn

# names of columns
data_columns = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']

# read from csv file
pima_data = pd.read_csv("pima-indians-diabetes-database.csv", header=None, names=data_columns)

# select 5 features from the chosen dataset
features = ['pregnant', 'glucose', 'bp', 'skin', 'insulin']
X = pima_data[features]
y = pima_data.label

# split test and trainsing data by 40% testing and 60% training
X_train, X_test, y_train, y_test = tts(X, y, test_size=0.40, train_size=0.60, random_state=0)

# fit model with training data and test after fitting
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train)
y_pred = logistic_regression.predict(X_test)
print(X_test)

# print confusion matrix and classification report
pima_conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", pima_conf_matrix)
print("Binary Classification Report:\n", classification_report(y_test, y_pred))

# display confusion matrix as heatmap
plt.title("Confusion Matrix Heatmap")
sn.heatmap(pima_conf_matrix, annot=True, fmt="d")
plt.savefig('conf_matrix')

# print precision score, recall score, and F score
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# plot out the ROC curve
logreg_roc_auc = roc_auc_score(y_test, y_pred)
fpr, tpr, thresholds = roc_curve(y_test, logistic_regression.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' %logreg_roc_auc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Reciever Operating Characteristic Curve')
plt.legend(loc='lower right')
plt.savefig('Log_ROC')
plt.show()