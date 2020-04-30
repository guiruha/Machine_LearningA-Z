# MUESTRE THOMSON

# Importamos los datos

df <- read.csv("/home/guillem/CursosML/machinelearning-az-master/datasets/Part 6 - Reinforcement Learning/Section 32 - Upper Confidence Bound (UCB)/Ads_CTR_Optimisation.csv")

# Probamos la seleccion random
N <- dim(df)[1]
d <- dim(df)[2]
ads_selected <- integer(0)
total_reward <- 0
for (n in 1:N) {
  ad <- sample(1:10, 1)
  ads_selected <- append(ads_selected, ad)
  reward <- df[n, ad]
  total_reward <- total_reward + reward
}

# Visualizamos los datos aleatorios
hist(ads_selected,
     col = "navy",
     main = "Histograma de anuncios",
     xlab = "Ads",
     ylab = "Número de veces que el anuncio se ha seleccionado")

# Implementamos el MUESTRE DE THOMSON
d <- dim(df)[2]
N <- dim(df)[1]
num_rewards_1 <-integer(d)
num_rewards_0 <- integer(d)

ads_selected <- integer(0)
total_reward <- 0
for(n in 1:N){
  max_random <- 0
  ad <- 0
  for(i in 1:d){
    random_beta <- rbeta(n = 1,
                         shape1 = num_rewards_1[i] + 1,
                         shape2 = num_rewards_0[i] + 1)
    
    if(random_beta > max_random){
      max_random <- random_beta
      ad <- i
    }
  }
  ads_selected = append(ads_selected, ad)
  reward = df[n, ad]
  if (reward == 1){
    num_rewards_1[ad] <- num_rewards_1[ad] + 1
  } else {
    num_rewards_0[ad] <- num_rewards_0[ad] + 1
  }
  total_reward <- total_reward + reward
}

# Visualizamos los resultados en un Histograma
hist(ads_selected,
     col = "lightblue",
     main = "Histograma de Anuncios",
     xlab = "ID del Anuncio",
     ylab = "Frecuencia de uso del anuncio")
