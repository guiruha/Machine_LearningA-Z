#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 12:21:36 2020

@author: guillem
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ARBOL DE REGRESIÓN

# Importamos los datos
df = pd.read_csv("/home/guillem/CursosML/machinelearning-az-master/datasets/Part 2 - Regression/Section 8 - Decision Tree Regression/Position_Salaries.csv")
X = df.iloc[:, 1:2]
y = df[['Salary']]

# Ajustamos el modelo de Arboles

from sklearn.tree import DecisionTreeRegressor
regtree = DecisionTreeRegressor(random_state = 0)

regtree.fit(X, y)

# Predecimos una muestra concreta
y_pred = regtree.predict(np.array(6.5).reshape(-1, 1))

# Visualizamos los resultados del Modelo De Arbol
fig, ax = plt.subplots(1, 1)
ax.scatter(X, y, color = "red")
ax.plot(X, regtree.predict(X), color = "blue")
plt.show()

# Visualizamos los resultados más "suavizado"
X_grid = np.arange(0, 10, 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
fig, ax = plt.subplots(1, 1)
ax.scatter(X, y, color = "red")
ax.plot(X_grid, regtree.predict(X_grid), color = "blue")
plt.show()


# BOSQUES ALEATORIOS

df = pd.read_csv("/home/guillem/CursosML/machinelearning-az-master/datasets/Part 2 - Regression/Section 9 - Random Forest Regression/Position_Salaries.csv")
X = df.iloc[:, 1:2]
y = df[['Salary']]

# Ajustamos el modelo Random Forest
from sklearn.ensemble import RandomForestRegressor
regrf = RandomForestRegressor(n_estimators = 500, random_state = 0)
regrf.fit(X, y)

# Predicción puntual

y_pred = regrf.predict(np.array(6.5).reshape(-1, 1))

# Visualizamos los resultados más "suavizado"

X_grid = np.arange(0, 10, 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
fig, ax = plt.subplots(1, 1, figsize = (15, 10))
ax.scatter(X, y, color = "red")
ax.plot(X_grid, regrf.predict(X_grid), color = "blue")
ax.annotate('Predicción', xy = (6.5, regrf.predict(np.array(6.5).reshape(-1, 1))),
            xytext = (6.5, 60000), arrowprops = dict(width = 0.1,facecolor = "black"))
plt.show()

