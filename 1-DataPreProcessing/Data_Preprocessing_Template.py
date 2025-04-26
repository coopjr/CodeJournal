# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 17:48:51 2025

@author: Sandeep
"""

#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split  # Correct import

# Importing Dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Splitting the dataset
X_train, X_test, y_train , y_test = train_test_split(X,y, test_size = 0.2, random_state = 9)

"""Feature scaling 
(a lot of ml models are based on eucledian distance) same scaled vars
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) # using same scaling parameters that were learned from the training data
"""





