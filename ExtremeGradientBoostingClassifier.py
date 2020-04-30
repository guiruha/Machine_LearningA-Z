#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 17:36:32 2020

@author: guillem
"""


# Extreme Gradient Boosting

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

# Importamos el dataset

df = pd.read_csv('/home/guillem/CursosML/machinelearning-az-master/datasets/Part 8 - Deep Learning/Section 39 - Artificial Neural Networks (ANN)/Churn_Modelling.csv')

X = df.iloc[:, 3:-1].values
y = df.iloc[:, -1].values

# Sacamos la variables dummys
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import make_column_transformer

lbencoder = LabelEncoder()
X[:, 1] = lbencoder.fit_transform(X[:, 1])
X[:, 2] = lbencoder.fit_transform(X[:, 2])

onehotencoder_x = make_column_transformer((OneHotEncoder(), [1]), remainder = "passthrough")
X = onehotencoder_x.fit_transform(X)
X = X[:, 1:] # Eliminamos una de las dummys para evitar multicolinealidad

# Sacamos el train y test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1997)

# Entrenamos el model XGBoost

xgbclas = xgb.XGBClassifier(n_jobs = -1)

xgbclas.fit(X_train, y_train)

# Hacemos un K-Fold crossvalidation (AÑANDIDO NO NECESARIO)
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(xgbclas, X = X_train, y = y_train, cv  = 10)

'La media de scores es {} con desviación típica de {}'.format(accuracies.mean(), accuracies.std())

# Predecimos los resultados
y_pred = xgbclas.predict(X_test)

# Comparamos la precisión de la red
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm

sum(diag(cm))/sum(cm)
