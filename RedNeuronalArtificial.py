#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 11:48:09 2020

@author: guillem
"""


# Redes Neuronales Artificiales

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# Escalamos las variables (MUY IMPORTANTE)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Creamos la estructura de la red neuronal artificial

import keras
from keras.models import Sequential
from keras.layers import Dense

rna = Sequential() # Inicializamos la red
rna.add(Dense(6, kernel_initializer = "uniform", activation = "relu", input_dim = 11)) # Empezamos añandiendo un layer de 6 nodos con funciónn de activación ReLU
# Inicializamos aleatorios los pesos con una distribución uniforme
rna.add(Dense(6, kernel_initializer = "uniform", activation = "relu"))
rna.add(Dense(1, kernel_initializer = "uniform", activation = "sigmoid")) # Capa de salida con función de activación sigmoide

# Realizamos la compilación de la red con un optimizador Adam y función de coste de crosentropia
rna.compile(optimizer = 'adam', loss = "binary_crossentropy", metrics = ['accuracy'])

# Entrenamos la red neuronal

rna.fit(X_train , y_train, batch_size = 10, epochs = 100)

# Predecimos los resultados
y_pred = rna.predict(X_test)
y_pred = (y_pred > 0.5).astype('int')

# Comparamos la precisión de la red
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

sum(diag(cm))/sum(cm)
