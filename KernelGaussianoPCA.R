# KERNEL PCA

df <- read.csv('~/CursosML/machinelearning-az-master/datasets/Part 9 - Dimensionality Reduction/Section 45 - Kernel PCA/Social_Network_Ads.csv')

df <- df[, 3:5]

library(caTools)
set.seed(123)
split <- sample.split(df$Purchased, SplitRatio = 0.8)
train_set <- subset(df, split == TRUE)
test_set <- subset(df, split == FALSE)

# Escalamos los valores
train_set[,1:2] <- scale(training_set[, 1:2])
test_set[,1:2] <- scale(test_set[, 1:2])

# Reducimos dimensión con Kernel PCA
library(kernlab)


kpca <- kpca(~., data = train_set[, -3], 
             kernel = 'rbfdot', features = 2)

train_set_pca <- as.data.frame(predict(kpca, train_set))
train_set_pca$Purchased <- train_set$Purchased

test_set_pca <- as.data.frame(predict(kpca, test_set))
test_set_pca$Purchased <- test_set$Purchased

# Ajustamos el modelo de regresión logística
clasreg <- glm(Purchased ~ ., 
               data = train_set_pca, 
               family = binomial)

# Predicción de los resultados
proba_pred <- predict(clasreg, type = "response",
                      newdata = test_set_pca[, -3])

y_pred <- ifelse(proba_pred > 0.5, 1, 0)
y_pred

# Crear una matriz de confusión
cm <- table(test_set[, 3], y_pred)

# Visualización de los datos
library(ElemStatLearn)
set <- train_set_pca
X1 <- seq(min(set[,1])-1, max(set[, 1]) + 1,  by = 0.01)
X2 <- seq(min(set[,2])-1, max(set[, 2]) + 1, by = 0.01)

grid_set <- expand.grid(X1, X2)
colnames(grid_set) = c('V1', 'V2')
prob_set <- predict(clasreg, type = "response", newdata = grid_set)
y_grid <- ifelse(prob_set > 0.5, 1, 0)
plot(set[, -3],
     main = 'Clasificación (Conjunto de Testing)',
     xlab = "CP1", ylab = "CP2",
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))