#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 12:22:54 2020

@author: guillem
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/home/guillem/CursosML/machinelearning-az-master/datasets/Part 1 - Data Preprocessing/Section 2 -------------------- Part 1 - Data Preprocessing --------------------/Data.csv')

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

#Imputamos los datos faltantes
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = "mean")
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Codificar datos categoricos (ordinales)
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
#X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

# Codificar daos categoricos BIEN
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
onehotencoder_x = make_column_transformer((OneHotEncoder(), [0]), remainder = "passthrough")
X = onehotencoder_x.fit_transform(X)

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Dividir datos entre train y test

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Escalado de variable
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
# OJO solo ajustamos con el train set y se utiliza tanto en train como en test
X_test = sc_X.transform(X_test)
# No escalamos la variable y porque estamos en un problema de clasificiación



