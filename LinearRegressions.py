#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 10:41:34 2020

@author: guillem
"""
# REGRESIÓN LINEAL SIMPLE

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/home/guillem/CursosML/machinelearning-az-master/datasets/Part 2 - Regression/Section 4 - Simple Linear Regression/Salary_Data.csv")
X = df['YearsExperience']
y = df['Salary']

# Vamos a dividir el dataset en Train y Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#Ahora procedemos a hacer una regresión lineal simple
from sklearn.linear_model import LinearRegression
lsreg = LinearRegression()
lsreg.fit(X_train.values.reshape(-1, 1), y_train)

# Predecimos los resultados del conjunto test
y_pred = lsreg.predict(X_test.values.reshape(-1, 1))
print('y_pred = ', y_pred, '\n\n')

# Evaluamos su R-Square

print("R-Squared of the Linear Regression = ", lsreg.score(X_test.values.reshape(-1, 1), y_test)*100, '%')

# Visualizamos los datos del entrenamiento
fig, ax = plt.subplots(1, 1)
plt.title("Visualización de datos de entrenamiento")
ax.scatter(X_train, y_train, color = "red", marker = "x")
ax.plot(X_train, lsreg.predict(X_train.values.reshape(-1, 1)), color = "blue")
ax.set_xlabel("Años Experiencia")
ax.set_ylabel("Sueldo (en $)")
plt.tight_layout()

# Visualizamos los datos de test
fig, ax = plt.subplots(1, 1)
plt.title("Visualización de datos de test")
ax.scatter(X_test, y_test, color = "cyan", marker = "x")
ax.plot(X_train, lsreg.predict(X_train.values.reshape(-1, 1)), color = "blue")
ax.set_xlabel("Años Experiencia")
ax.set_ylabel("Sueldo (en $)")
plt.tight_layout()

# REGRESIÓN LINEAL MÚLTIPLE

# Importamos el nuevo dataset
df2 = pd.read_csv("/home/guillem/CursosML/machinelearning-az-master/datasets/Part 2 - Regression/Section 5 - Multiple Linear Regression/50_Startups.csv")
X2 = df2.iloc[:, :-1].values
y2 = df2.iloc[:, -1:].values

# Procedemos a realizar la codificación de datos categoricos
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
onehotencoder_x = make_column_transformer((OneHotEncoder(), [3]), remainder = "passthrough")
X2 = onehotencoder_x.fit_transform(X2)
X2 = X2[:, 1:] # Eliminamos la primera variable dummy (la utilizamos como la variable referencia)

# Dividimos entre train y test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size = 0.2, random_state = 0)

# Fiteamos una regresión lineal multiple
from sklearn.linear_model import LinearRegression
lrmult = LinearRegression()

lrmult.fit(X_train, y_train)

# Ahora predecimos los resultados del conjunto testing
lrmult.predict(X_test)

# ¿Qué R-adj tiene?

lrmult.score(X_test, y_test)

# REGRESIÓN LINEAL: ELMININACIÓN HACIA ATRÁS

import statsmodels.regression.linear_model as sm
X3 = np.append(arr = np.ones((len(X2), 1)).astype(int), values = X2, axis = 1)
X3_opt = X3[:, [0, 1, 2, 3, 4, 5]].tolist()
alpha = 0.05

reg_ols = sm.OLS(endog = y2, exog = X3_opt).fit()

reg_ols.summary()

# Eliminamos la variable x2, es decir, la variable con el p-valor más alto

X3_opt = X3[:, [0, 1, 3, 4, 5]].tolist()

reg_ols = sm.OLS(endog = y2, exog = X3_opt).fit()

reg_ols.summary()

# Eliminamos las variable x1

X3_opt = X3[:, [0, 3, 4, 5]].tolist()

reg_ols = sm.OLS(endog = y2, exog = X3_opt).fit()

reg_ols.summary()

# Eliminamos la variable x2

X3_opt = X3[:, [0, 3, 5]].tolist()

reg_ols = sm.OLS(endog = y2, exog = X3_opt).fit()

reg_ols.summary()

# FUNCIONES RECOGIDAS DEL CURSO ML DE A-Z (No creación propia)
# Esto se puede automatizar a través de funciones:

def backwarElimination(x, y, alpha):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x.tolist()).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > alpha:
            for j in range(0, numVars -i):
                if (regressor_OLS.pvalues[j].astype(float)== maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x

X_modeled = backwarElimination(X3, y2, 0.05)

# SI queremos tener en cuenta también el R-adj

def backwardElimination(x, SL):    
    numVars = len(x[0])    
    temp = np.zeros((50,6)).astype(int)    
    for i in range(0, numVars):        
        regressor_OLS = sm.OLS(y, x.tolist()).fit()        
        maxVar = max(regressor_OLS.pvalues).astype(float)        
        adjR_before = regressor_OLS.rsquared_adj.astype(float)        
        if maxVar > SL:            
            for j in range(0, numVars - i):                
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):                    
                    temp[:,j] = x[:, j]                    
                    x = np.delete(x, j, 1)                    
                    tmp_regressor = sm.OLS(y, x.tolist()).fit()                    
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)                    
                    if (adjR_before >= adjR_after):                        
                        x_rollback = np.hstack((x, temp[:,[0,j]]))                        
                        x_rollback = np.delete(x_rollback, j, 1)     
                        print (regressor_OLS.summary())                        
                        return x_rollback                    
                    else:                        
                        continue    
    regressor_OLS.summary()    
    return x 


X_modeled = backwarElimination(X3, y2, 0.05)
X_modeled
