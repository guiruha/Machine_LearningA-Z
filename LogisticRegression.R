# REGRESIÓN LOGÍSTICA

# Importamos la base de datos

df <- read.csv("~/CursosML/machinelearning-az-master/datasets/Part 3 - Classification/Section 14 - Logistic Regression/Social_Network_Ads.csv")
df <- df[, 3:5]

library(caTools)
set.seed(123)
split <- sample.split(df$Purchased, SplitRatio = 0.8)
training_set <- subset(df, split == TRUE)
test_set <- subset(df, split == FALSE)

# Escalamos los valores
training_set[,1:2] <- scale(training_set[, 1:2])
test_set[, 1:2] <- scale(test_set[, 1:2])

# Ajustamos el modelo de regresión logística
clasreg <- glm(Purchased ~ ., 
               data = training_set, 
               family = binomial)

# Predicción de los resultados
proba_pred <- predict(clasreg, type = "response",
                      newdata = test_set[, 1:2])

y_pred <- ifelse(proba_pred > 0.5, 1, 0)
y_pred

# Crear una matriz de confusión
cm <- table(test_set[, 3], y_pred)

# Visualización de los datos
library(ElemStatLearn)
set <- training_set
X1 <- seq(min(set[,1])-1, max(set[, 1]) + 1,  by = 0.01)
X2 <- seq(min(set[,2])-1, max(set[, 2]) + 1, by = 0.01)

grid_set <- expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
prob_set <- predict(clasreg, type = "response", newdata = grid_set)
y_grid <- ifelse(prob_set > 0.5, 1, 0)
plot(set[, -3],
     main = 'Clasificación (Conjunto de Testing)',
     xlab = "Edad", ylab = "Sueldo Estimado",
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))