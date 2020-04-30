#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 11:36:38 2020

@author: guillem
"""

# RED NEURONAL CONVOLUCIONAL

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Creamos la CNN

cnn = Sequential()

# Añadimos la capa de Convolución

cnn.add(Conv2D(filters = 32,kernel_size = (3, 3), 
                      input_shape = (64, 64, 3), activation = "relu"))

# Añadimos la capa de Max Pooling

cnn.add(MaxPooling2D(pool_size = (2,2)))

# Una segunda capa de convolución y max pooling
cnn.add(Conv2D(filters = 32,kernel_size = (3, 3), activation = "relu"))

cnn.add(MaxPooling2D(pool_size = (2,2)))

# Añadimos la capa de Flattening

cnn.add(Flatten())

# Añadimos las capas de la red neuronal

cnn.add(Dense(units = 128, activation = "relu"))
cnn.add(Dense(units = 1, activation = "sigmoid"))

# Compilamos la CNN

cnn.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

# Entrenamos la red neuronal con las imagenes de entrenamiento
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory('/home/guillem/CursosML/machinelearning-az-master/datasets/Part 8 - Deep Learning/Section 40 - Convolutional Neural Networks (CNN)/dataset/training_set',
                                                    target_size=(64, 64),
                                                    batch_size=32,
                                                    class_mode='binary')

test_set = test_datagen.flow_from_directory('/home/guillem/CursosML/machinelearning-az-master/datasets/Part 8 - Deep Learning/Section 40 - Convolutional Neural Networks (CNN)/dataset/test_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')

# TIEMPO ESTIMADO DE 8 HORAS
cnn.fit_generator(train_set,
                        steps_per_epoch=8000,
                        epochs=25,
                        validation_data=test_set,
                        validation_steps=2000)
