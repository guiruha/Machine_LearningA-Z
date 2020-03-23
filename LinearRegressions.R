# REGRESIÓN LINEAL SIMPLE

# Leemos el fichero del curso
df <- read.csv("~/CursosML/machinelearning-az-master/datasets/Part 2 - Regression/Section 4 - Simple Linear Regression/Salary_Data.csv")

head(df, 6)

# Divididmos el df en Train y Test
library(caTools)
split <- sample.split(df$Salary, SplitRatio = 2/3)
training_set <- subset(df, split == TRUE)
test_set <- subset(df, split == FALSE)

# Ahora vamos a ajustar una regresión simple
lsreg <- lm(Salary~YearsExperience, data = training_set)
summary(lsreg)

# Realizamos las predicciones en testing

y_pred <- predict(lsreg, newdata = test_set)

# Visualizamos los resultados en el conjunto de entrenamiento
library(ggplot2)
ggplot(data = training_set) + 
  geom_point(aes(YearsExperience, Salary), colour = "red") +
  geom_line(aes(YearsExperience, predict(lsreg, newdata = training_set)), colour = "blue") +
  ggtitle("Sueldo respecto a los Años de Experiencia") +
  xlab("Años de Experiencia")+
  ylab("Salario")

# Visualizamos los resultados en el conjunto de test
ggplot(data = test_set)+
  geom_point(aes(YearsExperience, Salary), colour = "black")+
  geom_line(aes(YearsExperience, predict(lsreg, newdata = test_set)), colour = "navy")+
  ggtitle("Sueldo respecto a los Años de Experiencia") +
  xlab("Años de Experiencia")+
  ylab("Salario")

# REGRESIÓN LINEAL MÚLTIPLE

df2 <- read.csv("~/CursosML/machinelearning-az-master/datasets/Part 2 - Regression/Section 5 - Multiple Linear Regression/50_Startups.csv")

df2$State <- factor(df2$State,
                       levels = c("New York", "California", "Florida"),
                       labels = c(1, 2, 3))
# Dividimos entre train y test set
library(caTools)
split <- sample.split(df2$Profit, SplitRatio = 0.8)
training_set2 <- subset(df2, split == TRUE)
test_set2 <- subset(df2, split == FALSE)

# Podemos ajustar el modelo de forma simple:
lmreg <- lm(Profit ~ ., data = training_set2)
summary(lmreg)

# Predecimos los resultado del conjunto testing
y_pred <- predict(lmreg, newdata = test_set2)

# REGRESIÓN LINEAL MÚLTIPLE: ELIMINACIÓN HACIA ATRÁS
lmreg <- lm(Profit ~ R.D.Spend + Administration + 
            Marketing.Spend + State , 
            data = df2)

summary(lmreg)

# Eliminamos la columna de State (No es relevante al tener un pvalor > 0.05)
lmreg <- lm(Profit ~ R.D.Spend + Administration + 
            Marketing.Spend, 
            data = df2)

summary(lmreg)

# Ahora eliminamos la columna de Adminitration

lmreg <- lm(Profit ~ R.D.Spend + 
              Marketing.Spend, 
            data = df2)

coef(summary(lmreg))[,"Pr(>|t|)"]

# Podríamos eliminar la columna de Marketing.Spend pero considero que no es necesario 
# (Aunque si siguieramos estrictamente el método de eliminación hacia atrás deberíamos hacerlo)


backwardElimination <- function(x, alpha) {
  numVars = length(x)
  for (i in c(1:numVars)){
    regressor = lm(formula = Profit ~ ., data = x)
    maxVar = max(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"])
    if (maxVar > alpha){
      j = which(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"] == maxVar)
      x = x[, -j]
    }
    numVars = numVars - 1
  }
  return(summary(regressor))
}

backwardElimination(training_set2, 0.05)
