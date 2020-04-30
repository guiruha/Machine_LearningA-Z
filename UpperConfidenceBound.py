#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 11:05:32 2020

@author: guillem
"""

# UPPER CONFIDENCE BOUND (UCB)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Cargamos el Dataset

df = pd.read_csv("/home/guillem/CursosML/machinelearning-az-master/datasets/Part 6 - Reinforcement Learning/Section 32 - Upper Confidence Bound (UCB)/Ads_CTR_Optimisation.csv")

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

# Algoritmo de upper confidence bound
N = df.shape[0]
d = df.shape[1]

num_selection = [0] * d
sum_rewards = [0] * d
ads_selected  = []
total_reward = 0
for n in range(0, N):
    max_upper_bound = 0
    ad = 0
    for i in range(0, d):
        if(num_selection[i] > 0):
            average_reward = sum_rewards[i]/num_selection[i]
            delta_i = np.sqrt(3/2*np.log(n+1)/num_selection[i])
            upper_bound = average_reward + delta_i
        
        else:
            upper_bound = 1e400
            
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    num_selection[ad] = num_selection[ad] + 1
    reward = df.values[n, ad]
    sum_rewards[ad] = sum_rewards[ad] + reward
    total_reward = total_reward + reward
    
    
# Visualizamos con un histograma
plt.subplots(1, 1, figsize = (10, 10))
plt.hist(ads_selected)
plt.title("Histograma de anuncios")
plt.xlabel("ID del Anuncio")
plt.ylabel("Frecuencia de visualización del anuncio")
plt.show()