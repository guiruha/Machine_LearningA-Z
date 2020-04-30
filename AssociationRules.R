# APRIORI

# Leemos el Conjunto de Datos
library(arules)
df <- read.transactions('~/CursosML/machinelearning-az-master/datasets/Part 5 - Association Rule Learning/Section 28 - Apriori/Market_Basket_Optimisation.csv',
                        sep = ',', rm.duplicates = TRUE)
summary(df)
itemFrequencyPlot(df, top = 10)

# Entrenamos el Algoritmo de Apriori con 0.003 de support
rules <- apriori(data = df, 
                 parameter = list(support = round((3*7/7500), 3), confidence = 0.2))

# Visualizamos los resultados

inspect(sort(rules, by = 'lift')[1:10])

# Entrenamos el Algoritmo de Apriori con 0.004 de support
rules <- apriori(data = df, 
                 parameter = list(support = round((4*7/7500), 3), confidence = 0.2))

# Visualizamos los resultados de nuevo

inspect(sort(rules, by = 'lift')[1:10])

# ECLAT (APRIORI simplificado)

df <- read.transactions('~/CursosML/machinelearning-az-master/datasets/Part 5 - Association Rule Learning/Section 29 - Eclat/Market_Basket_Optimisation.csv',
                        sep = ',', rm.duplicates = TRUE)
summary(df)
itemFrequencyPlot(df, topN = 10)

# Entrenamos el Algoritmo Eclat con 0.004 de support

rules <- eclat(data = df, parameter = list(support = round((4*7/7500), 3), minlen = 2))
     
# Visualizamos resultados

inspect(sort(rules, by = 'support')[1:10])
