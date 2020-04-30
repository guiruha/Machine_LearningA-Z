# ÁRBOLES DE DECISIÓN

# Cargamos el dataset

df <- read.csv("~/CursosML/machinelearning-az-master/datasets/Part 2 - Regression/Section 8 - Decision Tree Regression/Position_Salaries.csv")

# Ajustamos el Modelo de Arbol de Decision
library(rpart)
regtree <- rpart(Salary ~ Level,
                 data = df)

# Predecimos una muestra concreta
predict(regtree, newdata = data.frame(Level = 6.5))

# Visualizamos el por qué de que vaya mal
library(ggplot2)
ggplot(data = df)+
  geom_point(aes(Level, Salary), colour = "red")+
  geom_line(aes(Level, predict(regtree, newdata = data.frame(Level = Level))), colour = "blue")

# Vamos a hacer un poco de hyperparameter tunning para solcionarlo

regtree <- rpart(Salary ~ Level,
                 data = df,
                 control = rpart.control(minsplit  = 1))

library(ggplot2)
ggplot(data = df)+
  geom_point(aes(Level, Salary), colour = "red")+
  geom_line(aes(Level, predict(regtree, newdata = data.frame(Level = Level))), colour = "blue")

x_grid <- seq(min(df$Level), max(df$Level), 0.1)

ggplot()+
  geom_point(aes(df$Level, df$Salary), colour = "red")+
  geom_line(aes(x_grid, predict(regtree, newdata = data.frame(Level = x_grid))), colour = "blue")

# BOSQUES ALEATORIOS

df <- read.csv("~/CursosML/machinelearning-az-master/datasets/Part 2 - Regression/Section 9 - Random Forest Regression/Position_Salaries.csv")

# Ajustamos el modelo de random forest

library(randomForest)
regrf <- randomForest(x = df[2], 
                      y = df$Salary,
                      ntree = 5000)
summary(regrf)

# Visualizamos los datos de forma ("suavizada")
library(ggplot2)
x_grid <- seq(min(df$Level), max(df$Level), 0.1)

ggplot()+
  geom_point(aes(df$Level, df$Salary), colour = "red")+
  geom_line(aes(x_grid, predict(regrf, newdata = data.frame(Level = x_grid))), colour = "blue")

