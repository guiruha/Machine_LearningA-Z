
# NATURAL LANGUAGE PROCESSING

# Importamos el dataset

df <- read.delim('/home/guillem/CursosML/machinelearning-az-master/datasets/Part 7 - Natural Language Processing/Section 36 - Natural Language Processing/Restaurant_Reviews.tsv', 
                 sep = '\t', quote = '', stringsAsFactors = FALSE)


# Limpiamos los textos

library(tm)
library(SnowballC)

corpus <- VCorpus(VectorSource(df$Review))
corpus <- tm_map(corpus, content_transformer(tolower))

# Consultamos el primer elemento del corpus
as.character(corpus[[1]])

corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removePunctuation)

as.character(corpus[[1]])

corpus <- tm_map(corpus, removeWords, stopwords(kind = 'en'))
# kind = 'en' procede de la libreria SnowballC

as.character(corpus[[1]])

# Stemizamos las palabras
corpus <- tm_map(corpus, stemDocument)

as.character(corpus[[1]])

corpus <- tm_map(corpus, stripWhitespace)

as.character(corpus[[1]])

# Creamos el bag of words
dtm <- DocumentTermMatrix(corpus)
dtm <- removeSparseTerms(dtm, 0.999)

dfmodel <- as.data.frame(as.matrix(dtm))

# Vamos a ajustar un RandomForest
dfmodel$Liked <- factor(df$Liked, levels = c(0, 1))

library(caTools)
set.seed(1997)

split <- sample.split(dfmodel$Liked, SplitRatio = 0.8)
train_set <- subset(dfmodel, split == TRUE)
test_set <- subset(dfmodel, split == FALSE)

library(randomForest)
rf <- randomForest(x = train_set[,-692],
                   y = train_set$Liked,
                   ntree = 10)

y_pred <- predict(rf, newdata = test_set[, -692])

# Comprobamos resutlados

cm <- table(test_set[, 692], y_pred)

paste0('Tenemos un score de: ', sum(diag(cm))/sum(cm))
