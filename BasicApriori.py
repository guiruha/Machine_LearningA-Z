#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 12:49:07 2020

@author: guillem
"""

# APRIORI

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("~/CursosML/machinelearning-az-master/datasets/Part 5 - Association Rule Learning/Section 28 - Apriori/Market_Basket_Optimisation.csv", header = None)
transactions = []

for i in range(0, len(df)):
    transactions.append([str(df.values[i, j]) for j in range(0, len(df.columns))])

transactions[:10]

# Importamos libreria local y entrenamos el apriori
from apyori import apriori

rules = apriori(transactions, 
                min_support = round((3*7/7500), 3), 
                min_confidence = 0.4, 
                min_length = 2,
                min_lift = 3)


# Visualización de resultados
results = list(rules)

dresults = pd.DataFrame(results)


for i in range(5):
    print('Relación {}'.format(i))
    print(results[i])
    print('='*25)
    
    