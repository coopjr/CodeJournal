# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 20:38:41 2025

@author: Sandeep
"""
#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split  # Correct import

# Importing Dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

#Splitting the dataset
X_train, X_test, y_train , y_test = train_test_split(X,y, test_size = 1/3, random_state = 0)

"""Feature scaling 
(a lot of ml models are based on eucledian distance) same scaled vars
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) # using same scaling parameters that were learned from the training data
"""

#Fitting Linear Regression moddel to Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()  
regressor.fit(X_train, y_train)

#predicting
y_pred = regressor.predict(X_test)

#Visualising training set
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train),color = 'blue')
plt.title('Salary vs Experoence (training set)')
plt.ylabel('Salary')
plt.xlabel('Years of Experience')
plt.show()

#Visualising testing set
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train),color = 'blue')
plt.title('Salary vs Experoence (training set)')
plt.ylabel('Salary')
plt.xlabel('Years of Experience')
plt.show()
