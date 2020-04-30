# Importamos los datos

df <- read.csv("~/CursosML/machinelearning-az-master/datasets/Part 2 - Regression/Section 7 - Support Vector Regression (SVR)/Position_Salaries.csv")

# Ajustamos el modelo SVR
library(e1071)
svr <- svm(Salary ~ Level, 
           data = df, 
           type = "eps-regression")

summary(svr)

# Predecimos los valores
y_pred <- predict(svr, newdata = data.frame(Level = 6.5))

# Visualizamos lo bien que se ajusta el modelo
library(ggplot2)
x_grid <- seq(min(df$Level), max(df$Level), 0.1)

ggplot()+
  geom_point(aes(df$Level, df$Salary), color = "red")+
  geom_line(aes(df$Level, predict(svr, newdata = data.frame(Level = df$Level)
                )),color = "blue") +
  ggtitle("Predicción con SVR")+
  xlab("Nivel de Empleado")+
  ylab("Sueldo (en $)")
            