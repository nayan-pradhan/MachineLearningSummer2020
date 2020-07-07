# In[25]:

import os
import numpy as np
import pandas as pd
from pandas import read_csv
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
data = pd.read_csv("student-mat", sep=";")

data = data[["G1", "G2", "G3"]]

predict = "G3"

X = np.array(data.drop([predict], 1)) # drop data for a column
y = np.array(data[predict]) # predicts for the previously droped data

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test) #accuracy
print(acc)

# linear = linear_model.LinearRegression()

# linear.fit(x_train, y_train)
# acc = linear.score(x_test, y_test)
# print(acc)

# print('Coefficient: \n', linear.coef_)
# print('Intercept: \n', linear.intercept_)

# predictions = linear.predict(x_test)

# for x in range(len(predictions)):
#     print(predictions[x], x_test[x], y_test[x])
