# LDA

# Importamos los datos
df <- read.csv("~/CursosML/machinelearning-az-master/datasets/Part 9 - Dimensionality Reduction/Section 44 - Linear Discriminant Analysis (LDA)/Wine.csv")

# Repartimos entre train y test
library(caTools)
set.seed(123)
split <- sample.split(df$Customer_Segment, SplitRatio = 0.8)
train_set <- subset(df, split == TRUE)
test_set <- subset(df, split == FALSE)

# Escalamos los valores
train_set[, -14] <- scale(train_set[, -14])
test_set[, -14] <- scale(test_set[, -14])

# Ajustamos con un LDA
library(MASS)
lda <- lda(formula = Customer_Segment ~ .,
           data = train_set)

train_set <- as.data.frame(predict(lda, train_set[,-14]))
test_set <- as.data.frame(predict(lda, test_set[,-14]))

# Permutamos las columnas
train_set <- train_set[, c(5, 6, 1)]
test_set <- test_set[, c(5, 6, 1)]

# Ajustamos el dataset con una SVM

svm <- svm(formula = class ~ .,
           data = train_set,
           type = "C-classification",
           kernel = "linear")

# Predicción de los resultados con el conjunto de testing
y_pred <- predict(svm, newdata = test_set[,-3])

# Creamos matriz de confusion

cm = table(test_set[, 3], y_pred)
cm

# Visualizamos los resultados

library(ElemStatLearn)
set <- train_set
X1 <- seq(min(set[,1])-1, max(set[, 1]) + 1,  by = 0.01)
X2 <- seq(min(set[,2])-1, max(set[, 2]) + 1, by = 0.01)

grid_set <- expand.grid(X1, X2)
colnames(grid_set) = c('x.LD1', 'x.LD2')
y_grid <- predict(svm, newdata = grid_set)
plot(set[, -3],
     main = 'SVM (Conjunto de Training)',
     xlab = "LD1", ylab = "LD2",
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 2, 'deepskyblue', ifelse(y_grid == 1, 'springgreen3', 'tomato')))
points(set, pch = 21, bg = ifelse(set[,3] == 2, 'blue3',ifelse(set[, 3] == 1, 'green4', 'red3')))