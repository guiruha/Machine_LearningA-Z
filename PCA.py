#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 11:20:41 2020

@author: guillem
"""


# PCA

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importamos el dataset

df = pd.read_csv('/home/guillem/CursosML/machinelearning-az-master/datasets/Part 9 - Dimensionality Reduction/Section 43 - Principal Component Analysis (PCA)/Wine.csv')

X = df.iloc[:, 0:13].values
y = df.iloc[:, 13].values

# Dividimos entre train y set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Escalamos las variables

from sklearn.preprocessing import StandardScaler 
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Reducimos la dimension con PCA/ACP
from sklearn.decomposition import PCA
pca = PCA(n_components = None) # Ya está None por defecto
X_train = pca.fit_transform(X_train)
explained_variance = pca.explained_variance_ratio_

# Visualizamos la suma acumulativa de varianza explicadas
fig, ax = plt.subplots(1, 1, figsize = (15, 10))
ax.plot(range(X_train.shape[1]), explained_variance.cumsum(), marker = '*', markersize = 30)
plt.xticks(range(X_train.shape[1]))
plt.tight_layout()

# Nos quedamos con 2 dimensiones
from sklearn.decomposition import PCA
pca = PCA(n_components = 2) 
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

# Combinamos con la regresión logistica
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state = 1997)
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

# Creamos una matriz de confusion
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Representación gráfica de los resultados del algoritmo
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.subplots(1, 1, figsize = (20, 20))
plt.contourf(X1, X2, lr.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Clasificador (conjunto de Entrenamiento)')
plt.xlabel('CP1')
plt.ylabel('CP2')
plt.legend()
plt.show()

# Representación del conjunto de test
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.subplots(1, 1, figsize = (20, 20))
plt.contourf(X1, X2, lr.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Clasificador (conjunto de Test)')
plt.xlabel('CP1')
plt.ylabel('CP2')
plt.legend()
plt.show()
