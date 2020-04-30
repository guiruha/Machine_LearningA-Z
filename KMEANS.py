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
df = pd.read_csv("/home/guillem/CursosML/machinelearning-az-master/datasets/Part 4 - Clustering/Section 24 - K-Means Clustering/Mall_Customers.csv")

X = df.iloc[:, [3, 4]].values

# Utilizamos el metodo de Kmeans
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = "k-means++", max_iter = 300, n_init = 20)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
fig, ax = plt.subplots(1, 1, figsize = (20, 13))
plt.plot(range(1, 11), wcss, color = "navy")
plt.title("Método del codo")
plt.xlabel("Nº de Clústers")
plt.ylabel("WSCC(k)")
plt.show()

# Aplicamos el K-means con el número optiomo de clusters
kmeans = KMeans(n_clusters = 5, init = "k-means++", max_iter = 300, n_init = 20, random_state = 0)
kmeans.fit(X)

y_kmeans = kmeans.predict(X)

# Visualizamos los clusters
colors = ["red", "blue", "orange", "purple", "green"]
fig, ax = plt.subplots(1, 1, figsize = (10, 10))
for i, c in enumerate(colors):    
    plt.scatter(X[y_kmeans == i,0], X[y_kmeans == i, 1], color = c, label = 'Clúster {}'.format(i))
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker = "*", s = 300, color = "black", label = "Baricentros")
plt.legend()
plt.title("Clústers de Clientes")
plt.xlabel("Ingresos anuales")
plt.ylabel("Puntuación de Gastos (1 -100)")
plt.show()

