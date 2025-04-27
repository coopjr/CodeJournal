# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 21:42:59 2025

@author: Sandeep
"""

#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split  # Correct import
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# Importing Dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Encoding Categorical Data
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3]) 
ct = ColumnTransformer(
    transformers=[
        ('encoder', OneHotEncoder(), [3])
    ],
    remainder='passthrough'
)
X = ct.fit_transform(X)

#Avoiding Dummy Variable Trap
X = X[:, 1:]

#Splitting the dataset
X_train, X_test, y_train , y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

'''#Feature scaling  usually libraries take care of this
#(a lot of ml models are based on eucledian distance) same scaled vars
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) # using same scaling parameters that were learned from the training data
'''
#Fitting Multiple Linear Regression to the training set
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#Predicting the Test set
y_pred = regressor.predict(X_test)

#Building the optimal model using Backward Elimination
''' y = b0X0 + b1X1 +...+ bnXn  making b0 *1 since stats models lib doesnt consider b0
 we add a column of 1s for b0'''
X = np.append(arr = np.ones((50 , 1)).astype(int), values = X , axis=1)

#Ordinary Least Squares minimizes SSE
X_opt = X[:, [0,1,2,3,4,5]]

X_opt = np.array(X_opt, dtype=float)
y = np.array(y, dtype=float)

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,1,3,4,5]]

X_opt = np.array(X_opt, dtype=float)
y = np.array(y, dtype=float)

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3,4,5]]

X_opt = np.array(X_opt, dtype=float)
y = np.array(y, dtype=float)

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3,5]]

X_opt = np.array(X_opt, dtype=float)
y = np.array(y, dtype=float)

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()