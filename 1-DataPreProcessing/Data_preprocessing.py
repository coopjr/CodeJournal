# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 20:38:41 2025

@author: Sandeep
"""
#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split  # Correct import
from sklearn.preprocessing import StandardScaler

# Importing Dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Handling Missing Values
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])

# Encoding Categorical Data
#labelencoder_X = LabelEncoder()
#X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

'''# One-hot encode column 0 manually
onehotencoder = OneHotEncoder(sparse_output=False)
X_categorical = onehotencoder.fit_transform(X[:, [0]])
X = np.concatenate((X_categorical, X[:, 1:]), axis=1)'''

# alternate way
ct = ColumnTransformer(
    transformers=[
        ('encoder', OneHotEncoder(), [0])
    ],
    remainder='passthrough'
)

X = ct.fit_transform(X)

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#Splitting the dataset
X_train, X_test, y_train , y_test = train_test_split(X,y, test_size = 0.2, random_state = 9)

#Feature scaling 
#(a lot of ml models are based on eucledian distance) same scaled vars
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) # using same scaling parameters that were learned from the training data






