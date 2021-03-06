# Random Forest Classifier

# Importamos la base de datos

df <- read.csv("~/CursosML/machinelearning-az-master/datasets/Part 3 - Classification/Section 20 - Random Forest Classification/Social_Network_Ads.csv")
df <- df[, 3:5]

#Codificamos la variable dependiente como factor
df$Purchased <- factor(df$Purchased, levels = c(0, 1))

library(caTools)
set.seed(123)
split <- sample.split(df$Purchased, SplitRatio = 0.8)
training_set <- subset(df, split == TRUE)
test_set <- subset(df, split == FALSE)

# Ajustamos el modelo de randomForest
library(randomForest)
rf <- randomForest(x = training_set[, -3],
                   y = training_set$Purchased,
                   ntree = 10)

# Predicción de los resultados
y_pred <- predict(rf, newdata = test_set[, -3], type = "class")

# Crear una matriz de confusión
cm <- table(test_set[, 3], y_pred)

# Visualización de los datos
library(ElemStatLearn)
set = training_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.05)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 150)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(rf, newdata = grid_set, type = "class")
plot(set[, -3],
     main = 'Árbol de Decisión (Conjunto de Entrenamiento)',
     xlab = 'Edad', ylab = 'Sueldo Estimado',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))


