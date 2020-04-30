# XGBoost

df <- read.csv('~/CursosML/machinelearning-az-master/datasets/Part 8 - Deep Learning/Section 39 - Artificial Neural Networks (ANN)/Churn_Modelling.csv')
df <- df[, 4:14]

# Sacamos las variables dummy
df$Geography <- as.numeric(factor(df$Geography,
                                  levels = c("France", "Spain", "Germany"),
                                  labels = c(1, 2, 3)))
df$Gender <- as.numeric(factor(df$Gender), 
                        levels = c("Male", "Female"),
                        labels = c(1, 2))

# Dividimos entre train y test
library(caTools)
set.seed(1997)
split <- sample.split(df$Exited, SplitRatio = 0.8)
train_set <-subset(df, split == TRUE)
test_set <- subset(df, split == FALSE)

# Entrenamos el model XGBoost

library(xgboost)

xgbclas <- xgboost(data = as.matrix(train_set[, -11]),
                   label = train_set$Exited,
                   nrounds = 10)

# Aplicamos k-fold cross validation (NO NECESARIO)
library(caret)
folds <- createFolds(train_set$Exited, k = 10)
cv <- lapply(folds, function(x){
      train_fold <- train_set[-x, ]
      test_fold <- train_set[x, ]
      xgbclas <- xgboost(data = as.matrix(train_fold[, -11]),
                         label = train_fold$Exited,
                         nrounds = 10)
      y_pred <- predict(xgbclas, newdata = as.matrix(test_fold[,-11]))
      y_pred <- (y_pred >= 0.5)
      cm <- table(test_fold[, 11], y_pred)
      accuracy <- sum(diag(cm))/sum(cm)
      return(accuracy)
})

mean(as.numeric(cv))
sd(as.numeric(cv))



