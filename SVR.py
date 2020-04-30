#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 11:18:42 2020

@author: guillem
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importamos el dataset

df = pd.read_csv("/home/guillem/CursosML/machinelearning-az-master/datasets/Part 2 - Regression/Section 7 - Support Vector Regression (SVR)/Position_Salaries.csv")
X = df.iloc[:, 1:2]
y = df[['Salary']]

# La SVR necesita que escalemos los datos
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()

X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# Ajustamos los datos en una SVR y vamos a utilizar el kernel gaussiano

from sklearn.svm import SVR
svr = SVR(kernel = "rbf")
svr.fit(X, y)

# Predecimos con la SVR
y_pred = svr.predict(X)

casodet = svr.predict(sc_X.transform(np.array([6.5]).reshape(-1, 1)))

sc_y.inverse_transform(casodet)

# Visualizamos la predicción
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
fig, ax = plt.subplots(1, 1)
ax.scatter(X, y, color = "red")
ax.plot(X_grid, svr.predict(X_grid), color = "blue")
plt.show()

# Visualizamos los datos sin escalar

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
fig, ax = plt.subplots(1, 1)
ax.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = "red")
ax.plot(sc_X.inverse_transform(X_grid), sc_y.inverse_transform(svr.predict(X_grid)), color = "blue")
plt.show()

