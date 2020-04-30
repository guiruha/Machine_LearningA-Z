# Clustering con KMEANS

# Importar los datos

df <- read.csv("/home/guillem/CursosML/machinelearning-az-master/datasets/Part 4 - Clustering/Section 24 - K-Means Clustering/Mall_Customers.csv")
X <- df[, 4:5]

# Usamos el dendorgrama para encontrar el número óptimo de clsuters
set.seed(6)
dendogram <- hclust(dist(X, method = "euclidean"),
                   method = "ward.D2")
plot(dendogram)


# Ajustamos el clustering jerárquico

hc <- hclust(dist(X, method = "euclidean"),
             method = "ward.D2")

y_hc <- cutree(hc, k=5)

# Visualizamos los clusters
library(cluster)

clusplot(X, 
         y_hc,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         label = 4,
         plotchar = FALSE,
         span = TRUE,
         main = "Clustering Clientes",
         xlab = "Ingresos anuales",
         ylab = "Puntuación de clientes")
