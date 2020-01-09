# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 12:29:32 2019

@author: shourt
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('manuf_delay1.csv')
X = dataset.iloc[:, :13].values
X1 = dataset.iloc[:, :13].values
y = dataset.iloc[:, -1].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X = LabelEncoder()

#dataset_encoded = dataset.apply(labelencoder_X.fit_transform)
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
X[:, 4] = labelencoder_X.fit_transform(X[:, 4])
X[:, 5] = labelencoder_X.fit_transform(X[:, 5])
X[:, 7] = labelencoder_X.fit_transform(X[:, 7])
X[:, 8] = labelencoder_X.fit_transform(X[:, 8])
X[:, 9] = labelencoder_X.fit_transform(X[:, 9])
X[:, 10] = labelencoder_X.fit_transform(X[:, 10])
X[:, 11] = labelencoder_X.fit_transform(X[:, 11])
X[:, 12] = labelencoder_X.fit_transform(X[:, 12])

onehotencoder = OneHotEncoder(categorical_features = 'all')
X = onehotencoder.fit_transform(X).toarray()

#train test split
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X, y)

y_pred = regressor.predict(X_test)
y_pred = np.rint(y_pred)

from sklearn.metrics import r2_score
r2_score(y_test,y_pred)
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, y_pred)

count = {}
count1 ={}
for i in y_pred:    
    count[i] = count.get(i, 0) + 1
    
for i in y_test:    
    count1[i] = count1.get(i, 0) + 1
#graphs
#comparison of actual and predicted values

plt.plot(count)
plt.plot(count1)
plt.show()

plt.plot(y_test[1500:1600],color = "red")
plt.plot(y_pred[1500:1600],color = "blue")
plt.show()

plt.scatter(X1[:, 1], y, color = 'red')
plt.title('manufacturing delay- L-EMC')
plt.xlabel('Plant')
plt.ylabel('Days')
plt.show()