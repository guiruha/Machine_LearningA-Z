# UPPER CONFIDENCE BOUND

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

# Implementamos la UCB
d <- dim(df)[2]
N <- dim(df)[1]
num_selections <- integer(d)
sum_rewards <- integer(d)
ads_selected <- integer(N)
  
number_of_selections = integer(d)
sums_of_rewards = integer(d)
ads_selected = integer(0)
total_reward = 0
for(n in 1:N){
  max_upper_bound = 0
  ad = 0
  for(i in 1:d){
    if(number_of_selections[i]>0){
      average_reward = sums_of_rewards[i] / number_of_selections[i]
      delta_i = sqrt(3/2*log(n)/number_of_selections[i])
      upper_bound = average_reward + delta_i
    }else{
      upper_bound = 1e400
    }
    if(upper_bound > max_upper_bound){
      max_upper_bound = upper_bound
      ad = i
    }
  }
  ads_selected = append(ads_selected, ad)
  number_of_selections[ad] = number_of_selections[ad] + 1
  reward = df[n, ad]
  sums_of_rewards[ad] = sums_of_rewards[ad] + reward
  total_reward = total_reward + reward
}

# Visualizamos los resultados en un Histograma
hist(ads_selected,
     col = "lightblue",
     main = "Histograma de Anuncios",
     xlab = "ID del Anuncio",
     ylab = "Frecuencia de uso del anuncio")