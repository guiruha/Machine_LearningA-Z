#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 11:27:31 2020

@author: guillem
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importamos la librerías
df = pd.read_csv("/home/guillem/CursosML/machinelearning-az-master/datasets/Part 4 - Clustering/Section 25 - Hierarchical Clustering/Mall_Customers.csv")

X = df.iloc[:, [3, 4]].values

# Utilizamos un dendograma para encontrar el número de clusters
import scipy.cluster.hierarchy as sch
plt.subplots(1, 1, figsize = (15, 10))
dendogram = sch.dendrogram(sch.linkage(X, method = "ward", metric = "euclidean"))
plt.title("Dendrogram")
plt.xlabel("Clientes")
plt.ylabel("Distancia Euclídea")

# Ajustamos el clustering jerárquico a nuestro conjunto de datos
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = "euclidean", linkage = "ward")
y_hc = hc.fit_predict(X)

# Visualización de los clsuters
colors = ["red", "blue", "orange", "purple", "green"]
fig, ax = plt.subplots(1, 1, figsize = (10, 10))
for i, c in enumerate(colors):    
    plt.scatter(X[y_hc == i,0], X[y_hc == i, 1], color = c, label = 'Clúster {}'.format(i))
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker = "*", s = 200, color = "black", label = "Baricentros")
plt.legend()
plt.title("Clústers de Clientes")
plt.xlabel("Ingresos anuales")
plt.ylabel("Puntuación de Gastos (1 -100)")
plt.show()
