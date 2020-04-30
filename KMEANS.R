# Clustering con KMEANS

# Importar los datos

df <- read.csv("/home/guillem/CursosML/machinelearning-az-master/datasets/Part 4 - Clustering/Section 25 - Hierarchical Clustering/Mall_Customers.csv")
X <- df[, 4:5]

# Usamos la tecnica del codo con Kmeans
set.seed(6)
wcss <- c()
for (i in 1:10){
  wcss[i] <- sum(kmeans(X, i)$withinss)
}
plot(1:10, wcss, "b", main = "Método del codo",
     xlab = "Nº de Clusters (k)", ylab = "WCSS(k)")

# Aplicamos el algoritmo de k-means con el k óptimo
set.seed(29)
kmeans <- kmeans(X, 5, iter.max = 300, nstart = 10)

# Visualización de los clusters
library(cluster)

clusplot(X, 
         kmeans$cluster,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         label = 4,
         plotchar = FALSE,
         span = TRUE,
         main = "Clustering Clientes",
         xlab = "Ingresos anuales",
         ylabl = "Puntuación de clientes")