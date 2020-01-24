import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

data = pd.read_csv('../Emotion_features.csv')
data2 = pd.read_csv('Emotion_features.csv')
Z=data2.ix[:, 'tempo':]
X = data.ix[:, 'tempo':]
y = data['class']
featureName = list(X)
for name in featureName:
    X[name] = (X[name]-X[name].min())/(X[name].max()-X[name].min())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.54 , random_state=26)

knn = KNeighborsClassifier()
param_grid = { 'n_neighbors': np.arange(1, 30) }
knn_cv = GridSearchCV(knn, param_grid, cv=10)
knn_cv.fit(X_train, y_train)

print(knn_cv.best_params_)
print("Baseline Accuracy: "),
print(knn_cv.best_score_)

y_pred = knn_cv.predict(Z)
cm=confusion_matrix(y_test, y_pred)
f, ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_test")
print ("""
       ---  KNN Confusion Matrix visualization ---
        """)

plt.show()
print(y_pred)
