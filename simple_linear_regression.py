# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 22:07:53 2017

@author: Saikiran Avula
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 


#importing data

data_set = pd.read_csv('Salary_Data.csv')
x = data_set.iloc[: , :-1].values
y = data_set.iloc[: , 1].values


#Spliting the dataset into training and test set.

from sklearn.cross_validation import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)



# Fitting simple linear regression to training set.

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train)


#Predicting the test set results.

y_pred = regressor.predict(x_test)


#Visualizing the training set results.

plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (training set)')
plt.xlabel('Year of experience')
plt.ylabel('salary in USD')
plt.show()


#Visualizing the test set results.

plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (test set)')
plt.xlabel('Year of experience')
plt.ylabel('salary in USD')
plt.show()


