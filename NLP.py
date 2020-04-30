#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 10:39:14 2020

@author: guillem
"""


# NATURAL LANGUAGE PROCESSING

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importamos el Dataset

df = pd.read_csv("/home/guillem/CursosML/machinelearning-az-master/datasets/Part 7 - Natural Language Processing/Section 36 - Natural Language Processing/Restaurant_Reviews.tsv", delimiter = "\t", quoting = 3)

df.head()

# Vamos a realizar la limpieza de texto con regex (expresiones regulares)
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# EJEMPLO CON UN SOLO REGISTRO
review = re.sub('[^a-zA-Z]', ' ', df['Review'][0])
review = review.lower()
review = review.split()
ps = PorterStemmer()
review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
review = ' '.join(review)

# Limpieza general del texto
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []

for i in range(df.shape[0]):
    review = re.sub('[^a-zA-Z]', ' ', df['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
# Realizamos la Bag of Words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = df['Liked']

# Ajustemos un Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1997)


rf = RandomForestClassifier()
rf.fit(X_train,y_train)

# Comprobamos los resultados
from sklearn.metrics import confusion_matrix

y_pred = rf.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

cm

'Random Forest tiene un precisión de {}'.format(sum(diag(cm))/sum(cm))

# Ajustemos un Naive Bayes
from sklearn.naive_bayes import GaussianNB
naive = GaussianNB()
naive.fit(X_train, y_train)

# Comprobamos los resultados
y_pred2 = naive.predict(X_test)

cm2 = confusion_matrix(y_test, y_pred2)

cm2

'Naive Bayes tiene un precisión de {}'.format(sum(diag(cm2))/sum(cm2))

# Por lo que RandomForest es preferible