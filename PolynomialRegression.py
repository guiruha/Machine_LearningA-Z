#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 11:25:11 2020

@author: guillem
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Leemos los datos￼
df = pd.read_csv("/home/guillem/CursosML/machinelearning-az-master/datasets/Part 2 - Regression/Section 6 - Polynomial Regression/Position_Salaries.csv")

fig, ax = plt.subplots(1, 1)
ax.scatter(df['Level'], df['Salary'])
plt.tight_layout()
# Como observamos la relación entre salario y level no es lineal
# Tiene forma de parábola

X = df[['Level']].values
y = df.loc[:, 'Salary'].values

# No dividiremos en este caso porque hay muy pocas muestras
# Ajustamos la regresión polinómica y una lineal para comparar

from sklearn.linear_model import LinearRegression

lsreg = LinearRegression()
lsreg.fit(X, y)

from sklearn.preprocessing import PolynomialFeatures

psfeat = PolynomialFeatures(degree = 2)

X_poly = psfeat.fit_transform(X)

psreg = LinearRegression()

psreg.fit(X_poly, y)

# Ajustamos con 3 grados
psfeat3 = PolynomialFeatures(degree = 3)

X_poly3 = psfeat3.fit_transform(X)

psreg3 = LinearRegression()

psreg3.fit(X_poly3, y)

# Ahora visualizamos una comparativa

fig, ax = plt.subplots(3, 1, figsize = (15, 10))
ax[0].scatter(X, y, color = "navy")
ax[1].scatter(X, y, color = "navy")
ax[2].scatter(X, y, color = "navy")
ax[0].plot(X, lsreg.predict(X), color = "red", linewidth = 3)
ax[1].plot(X, psreg.predict(X_poly), color = "red", linewidth = 3)
ax[2].plot(X, psreg3.predict(X_poly3), color = "red", linewidth = 3)
plt.tight_layout()














