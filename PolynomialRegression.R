# Leemos los datos

df <- read.csv("~/CursosML/machinelearning-az-master/datasets/Part 2 - Regression/Section 6 - Polynomial Regression/Position_Salaries.csv")

library(ggplot2)
ggplot(df, aes(Level, Salary))+
  geom_point(colour = 'red') +
  geom_smooth(se = FALSE)


# Ajustar un Modelo de Regresión Lineal con el Conjunto de Datos

lreg <- lm(Salary ~ Level, 
          data = df)
summary(lreg)

df$Level2 <- df$Level ** 2
plreg <- lm(Salary ~ Level + Level2,
            data = df)

summary(plreg)

df$Level3 <- df$Level**3
plreg3 <- lm(Salary ~ Level + Level2 + Level3,
             data = df)
summary(plreg3)

# Vamos a dibujar los resultados

ggplot(data = df)+
  geom_point(aes(Level, Salary),
             colour = "red", size = 2) +
  geom_line(aes(Level, predict(lreg, newdata = df)))+
  ggtitle('Regresión Lineal')

ggplot(data = df)+
  geom_point(aes(Level, Salary),
             colour = "red", size = 2) +
  geom_line(aes(Level, predict(plreg, newdata = df)))+
  ggtitle('Regresión Polinómica de Grado 2')

ggplot(data = df)+
  geom_point(aes(Level, Salary),
             colour = "red", size = 2) +
  geom_line(aes(Level, predict(plreg3, newdata = df))) +
  ggtitle('Regresión Polinómica de Grado 3')

# Predicción de resultados con regresión lineal
y_pred <- predict(lreg, newdata = data.frame(Level = 6.5))

# Predicción de resultados con regresión polinómica
y_predp <- predict(plreg, newdata = data.frame(Level = 6.5, Level2 = 6.5 ** 2))

# Vamos a hacer un gráfico más suave
x_grid = seq(min(df$Level), max(df$Level), 0.1)

ggplot() +
  geom_point(aes(x = df$Level , y = df$Salary),
             color = "red") +
  geom_line(aes(x = x_grid, y = predict(plreg3, 
                                        newdata = data.frame(Level = x_grid,
                                                             Level2 = x_grid ** 2,
                                                             Level3 = x_grid ** 3))),
            color = "blue") +
  ggtitle("Predicción (Modelo de Regresión)") +
  xlab("Nivel del empleado") +
  ylab("Sueldo (en $)")