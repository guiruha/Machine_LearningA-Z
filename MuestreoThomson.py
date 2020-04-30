#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 13:08:18 2020

@author: guillem
"""

# MUESTREO THOMSON 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Cargamos el Dataset

df = pd.read_csv("/home/guillem/CursosML/machinelearning-az-master/datasets/Part 6 - Reinforcement Learning/Section 33 - Thompson Sampling/Ads_CTR_Optimisation.csv")

# Implementamos una selección aleatoria
import random
N = 10000
d = 10
ads_selected = []
total_reward = 0
for n in range(0, N):
    ad = random.randrange(d)
    ads_selected.append(ad)
    reward = df.values[n, ad]
    total_reward = total_reward + reward
    
# Visualizamos un histograma de los resultados

plt.subplots(1, 1, figsize = (10 , 10))
plt.hist(ads_selected)
plt.title("Histograma de los anuncios seleccionados")
plt.xlablel("Anuncios")
plt.ylabel("Numero de veces que ha sido seleccionado")
plt.show()

# Vamos  intentar superar el baseline de arriba

# Algoritmo de MUESTREO THOMSON
N = df.shape[0]
d = df.shape[1]

num_reward_1 = [0] * d 
num_reward_0 = [0] * d

ads_selected  = []
total_reward = 0

import random
for n in range(0, N):
    max_random = 0
    ad = 0
    for i in range(0, d):
        random_beta = random.betavariate(num_reward_1[i]+1, num_reward_0[i]+1)
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    reward = df.values[n, ad]
    if reward == 1:
        num_reward_1[ad] = num_reward_1[ad] + 1
    else:
        num_reward_0[ad] = num_reward_0[ad] + 1
    total_reward = total_reward + reward
    
    
# Visualizamos con un histograma
plt.subplots(1, 1, figsize = (10, 10))
plt.hist(ads_selected)
plt.title("Histograma de anuncios")
plt.xlabel("ID del Anuncio")
plt.ylabel("Frecuencia de visualización del anuncio")
plt.show()
