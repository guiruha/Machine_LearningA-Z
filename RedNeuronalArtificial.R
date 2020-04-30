# Red Neuronal Artificial

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

# Escalamos los valores
train_set[, -11] <- scale(train_set[, -11])
test_set[, -11] <- scale(test_set[, -11])

# Creamos la red neuronal
# Neuralnet para regresión y H2O para general
library(h2o)

h2o.init(nthreads = -1)
rna <- h2o.deeplearning(y = "Exited",
                        training_frame = as.h2o(train_set),
                        activation = "Rectifier",
                        hidden = c(6, 6),
                        epochs = 100,
                        train_samples_per_iteration = -2)

# Predecimos los resultados
proba_pred <- h2o.predict(rna, newdata = as.h2o(test_set[, -11]))

y_pred <- ifelse(proba_pred > 0.5, 1, 0)
y_pred <- as.vector(y_pred)
# Creamos la matriz de confusión

cm <- table(test_set[, 11], y_pred)
cm

sum(diag(cm))/sum(cm)

# Cerramos la sesión de h2o

h2o.shutdown()

