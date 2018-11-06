library(lightgbm)
library(tidyverse)
library(caret)
library(xgboost)



validacion.f <- 1
ruido.skewed <- 0
n.round.sin.valid <- 10
split.valid <- 0.8
n.valores <- 1000

# algunas funciones
funciones <- list(
  sigmoide =  function(x) {
    100 / (1 + exp(-x/15))
  },
  seno = function(x) {
    100*sin(x/10)
  },
  sig_sen = function(x) {
    50 / (1 + exp(-x/15)) + 5*sin(x/10)
  }
)

rmse <- function(error) sqrt(mean(error^2, na.rm = TRUE))

# Construccion datos: real
datos_reales <- data.frame(x = runif(n.valores, 1,100))
datos_reales$y <- funciones[[1]](datos_reales$x)

ggplot(datos_reales) + geom_point(aes(x,y))

# dataset con ruido y anomalias
dataset <- datos_reales
if (ruido.skewed==0) {
  dataset$y <- datos_reales$y + rnorm(n.valores, sd=2)
} else {
  dataset$y <- datos_reales$y + (mean(rbeta(n.valores, 2,8))/2-rbeta(n.valores, 2,8))*20
}


# anomalias
n.anomalias <- 10
tamano.anomalia <- 10
fil.anomalias <- sample(1:nrow(datos_reales), n.anomalias)
dataset[fil.anomalias,'y'] <- dataset[fil.anomalias,'y'] + runif(n.anomalias, -tamano.anomalia, tamano.anomalia)

ggplot(dataset) + geom_point(aes(x,y))

tipo.mod <- 'regression_l2'

#### Parametros ####

param_lgb.df <- read.table('param_lgb.csv', sep = ';', dec = '.', header = TRUE, stringsAsFactors = FALSE)
param_lgb.df
row.names(param_lgb.df) <- param_lgb.df$tipo.mod
param_lgb.df$tipo.mod <- NULL

param_lgb <- as.list(as.data.frame(param_lgb.df['regression_l2',]))

param_lgb <- c(param_lgb,
               lambda_l2 = 100,
               num_leaves =50,
               # max_depth =6,
               learning_rate = 0.1025,
               num_threads = 10)




#### Split Validacion ####

train.i <- createDataPartition(dataset$y, p=split.valid, list = FALSE)
train <- dataset[train.i,]
valid <- dataset[-train.i,]

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#
################# LightGBM ##################################################
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

#### Dataset ####

dtrain <- lgb.Dataset(as.matrix(train$x), label = as.matrix(train$y))
dvalidation <- lgb.Dataset.create.valid(dtrain, as.matrix(valid$x), label = as.matrix(valid$y))

valids <- list(train = dtrain, eval = dvalidation)

#### Training ####

if(validacion.f == 1) {
  lgb.model <- lgb.train(param_lgb, data = dtrain,
                         valids,
                         # categorical_feature = col.categorical,
                         nrounds = 10000,
                         early_stopping_round = 5,#5,
                         alpha=0,
                         verbose = 1, eval_freq = 50)
} else {
  lgb.model <- lgb.train(param_lgb, data = lgb.Dataset(as.matrix(dataset$x), label = as.matrix(dataset$y)),
                         # categorical_feature = col.categorical,
                         nrounds = n.round.sin.valid,
                         # early_stopping_round = 5,
                         verbose = 1, eval_freq = 50)
}

#### Prediccion ####

pred.iter <- lgb.model$best_iter
lgb.model$best_iter


prediccion <- predict(lgb.model, as.matrix(train$x), num_iteration = pred.iter)
cat('RMSE train: ', rmse(train$y-prediccion),
    '\nRMSE test: ', rmse(valid$y-predict(lgb.model, as.matrix(valid$x), num_iteration = pred.iter)),
    '\nSD target: ', sd(train$y))

dataset.train = data.frame(x=train$x, pred = prediccion)

g<- ggplot(dataset) + geom_point(aes(x,y), size = 0.5) +
  geom_line(data = datos_reales, aes(x,y), color = 'blue', size = 1) +
  geom_line(data = dataset.train, aes(x, pred), col = 'red', size = 2)
if( (n.anomalias !=0) & (tamano.anomalia !=0) ) {
  g + geom_point(data = dataset[fil.anomalias,], aes(x,y), col = 'blue', size = 2)
} else {
  g
}


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#
################# LGBM quantile ####################################
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

param_lgb_q90 <- param_lgb
param_lgb_q90$objective <- 'quantile'
param_lgb_q90$metric <- 'quantile'
param_lgb_q90$alpha <- 0.9

if(validacion.f == 1) {
  lgb.model_q90 <- lgb.train(param_lgb_q90, data = dtrain,
                         valids,
                         # categorical_feature = col.categorical,
                         nrounds = 10000,
                         early_stopping_round = 5,#5,
                         alpha=0,
                         verbose = 1, eval_freq = 50)
} else {
  lgb.model_q90 <- lgb.train(param_lgb_q90, data = lgb.Dataset(as.matrix(dataset$x), label = as.matrix(dataset$y)),
                         # categorical_feature = col.categorical,
                         nrounds = n.round.sin.valid,
                         # early_stopping_round = 5,
                         verbose = 1, eval_freq = 50)
}

#### Prediccion ####

pred.iter_q90 <- lgb.model_q90$best_iter
lgb.model_q90$best_iter


prediccion_q90 <- predict(lgb.model_q90, as.matrix(train$x), num_iteration = pred.iter_q90)
cat('RMSE train: ', rmse(train$y-prediccion_q90),
    '\nRMSE test: ', rmse(valid$y-predict(lgb.model_q90, as.matrix(valid$x), num_iteration = pred.iter_q90)),
    '\nSD target: ', sd(train$y))


param_lgb_q10 <- param_lgb
param_lgb_q10$objective <- 'quantile'
param_lgb_q10$metric <- 'quantile'
param_lgb_q10$alpha <- 0.1

if(validacion.f == 1) {
  lgb.model_q10 <- lgb.train(param_lgb_q10, data = dtrain,
                             valids,
                             # categorical_feature = col.categorical,
                             nrounds = 10000,
                             early_stopping_round = 5,#5,
                             alpha=0,
                             verbose = 1, eval_freq = 50)
} else {
  lgb.model_q10 <- lgb.train(param_lgb_q10, data = lgb.Dataset(as.matrix(dataset$x), label = as.matrix(dataset$y)),
                             # categorical_feature = col.categorical,
                             nrounds = n.round.sin.valid,
                             # early_stopping_round = 5,
                             verbose = 1, eval_freq = 50)
}

#### Prediccion ####

pred.iter_q10 <- lgb.model_q10$best_iter
lgb.model_q10$best_iter


prediccion_q10 <- predict(lgb.model_q10, as.matrix(train$x), num_iteration = pred.iter_q10)
cat('RMSE train: ', rmse(train$y-prediccion_q10),
    '\nRMSE test: ', rmse(valid$y-predict(lgb.model_q10, as.matrix(valid$x), num_iteration = pred.iter_q10)),
    '\nSD target: ', sd(train$y))

dataset.train = data.frame(x=train$x, pred = prediccion, pred_q90 = prediccion_q90, pred_q10 = prediccion_q10)

g<- ggplot(dataset) + geom_point(aes(x,y), size = 0.5) +
  geom_line(data = datos_reales, aes(x,y), color = 'blue', size = 1) +
  geom_line(data = dataset.train, aes(x, pred), col = 'red', size = 2) +
  geom_line(data = dataset.train, aes(x, pred_q90), col = 'green', size = 2) +
  geom_line(data = dataset.train, aes(x, pred_q10), col = 'green', size = 2)
if( (n.anomalias !=0) & (tamano.anomalia !=0) ) {
  g + geom_point(data = dataset[fil.anomalias,], aes(x,y), col = 'blue', size = 2)
} else {
  g
}



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#
################# XGBoost ####################################
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

#### Parametros ####

param_xgb.df <- read.table('param_xgb.csv', sep = ';', dec = '.', header = TRUE, stringsAsFactors = FALSE)
row.names(param_xgb.df) <- param_xgb.df$tipo.mod
param_xgb.df$tipo.mod <- NULL

param_xgb <- as.list(as.data.frame(param_xgb.df['regression_l2',]))

param_xgb <- c(param_xgb,
               max_depth =5,
               learning_rate = 0.1025,
               nthread = 10)


#### Dataset ####

dtrain <- xgb.DMatrix(as.matrix(train$x), label = train$y)
dvalidation <- xgb.DMatrix(matrix(valid$x), label = valid$y)

wl <- list(train = dtrain, eval = dvalidation)

#### Training ####

n.round.sin.valid<-10 # 7000 para overfitting

if(validacion.f == 1) {
  xgb.model <- xgb.train(param_xgb, data = dtrain,
                         watchlist = wl,
                         nrounds = 10000,
                         early_stopping_rounds = 5,
                         verbose = 1, print_every_n = 50)
} else {
  xgb.model <- xgb.train(param_xgb, data = xgb.DMatrix(as.matrix(dataset$x), label = dataset$y),
                         nrounds = n.round.sin.valid,
                         watchlist = list(train = dtrain),
                         verbose = 1, print_every_n = 50)
}

#### Prediccion ####

pred.iter <- xgb.model$best_iter
xgb.model$best_iter


prediccion <- predict(xgb.model, as.matrix(train$x), num_iteration = pred.iter)
cat('RMSE train: ', rmse(train$y-prediccion),
    '\nRMSE test: ', rmse(valid$y-predict(xgb.model, as.matrix(valid$x), num_iteration = pred.iter)),
    '\nSD target: ', sd(train$y))

dataset.train = data.frame(x=train$x, pred = prediccion)

g<- ggplot(dataset) + geom_point(aes(x,y), size = 0.5) +
  geom_line(data = datos_reales, aes(x,y), color = 'blue', size = 1) +
  geom_line(data = dataset.train, aes(x, pred), col = 'red', size = 2)
if( (n.anomalias !=0) & (tamano.anomalia !=0) ) {
  g + geom_point(data = dataset[fil.anomalias,], aes(x,y), col = 'blue', size = 2)
} else {
  g
}




# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#
################# Binning e Interpolación ####################################
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


bin_size <- 5
dataset$x_bin <- round(dataset$x/bin_size,0)*bin_size

tabla_bines <- dataset %>% group_by(x_bin) %>% summarise(y_med = mean(y))
fit.func <- approxfun(tabla_bines$x_bin, tabla_bines$y_med, yleft = 0, yright = 100)
dataset$y_pred_bin <- fit.func(dataset$x)

g<- ggplot(dataset) + geom_point(aes(x,y), size = 0.5) +
  geom_line(data = datos_reales, aes(x,y), color = 'blue', size = 1) +
  geom_line(data = dataset, aes(x, y_pred_bin), col = 'red', size = 2)
if( (n.anomalias !=0) & (tamano.anomalia !=0) ) {
  g + geom_point(data = dataset[fil.anomalias,], aes(x,y), col = 'blue', size = 2)
} else {
  g
}


cat('RMSE: ', rmse(dataset$y-dataset$y_pred_bin),
    '\nSD target: ', sd(train$y))


#### Tensorflow ####


library(keras)
# install_keras()

use_implementation("tensorflow")
use_session_with_seed(7777, disable_gpu = FALSE, disable_parallel_cpu = FALSE)

mean <- apply(dataset %>% select(x), 2, mean)
std <- apply(dataset %>% select(x), 2, sd)
train_norm <- scale(dataset %>% select(x), center = mean, scale = std)
# valid_norm <- scale(dataset %>% select(x), center = mean, scale = std)


build_model <- function() {
  model <- keras_model_sequential() %>%
    layer_dense(units = 64, activation = "relu",
                input_shape = dim(train_norm)[[2]]) %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 1)
  model %>% compile(
    optimizer = "rmsprop",
    loss = "mse",
    metrics = c("mae")
  )
}

# k <- 4
# indices <- sample(1:nrow(train_norm))
# folds <- cut(indices, breaks = k, labels = FALSE)
# num_epochs <- 20
# all_scores <- c()
# for (i in 1:k) {
#   cat("processing fold #", i, "\n")
#   val_indices <- which(folds == i, arr.ind = TRUE)
#   val_data <- train_norm[val_indices,]
#   val_targets <- dataset$y[val_indices]
#   partial_train_data <- train_norm[-val_indices,]
#   partial_train_targets <- dataset$y[-val_indices]
#   model <- build_model()
#   model %>% fit(partial_train_data, partial_train_targets,
#                 epochs = num_epochs, batch_size = 1, verbose = 1)
#   results <- model %>% evaluate(val_data, val_targets, verbose = 1)
#   all_scores <- c(all_scores, results$mean_absolute_error)
# }

model <- build_model()
model %>% fit(train_norm, dataset$y,
              epochs = 500, batch_size = 1, verbose = 1)

model %>% save_model_hdf5('model_e500_b1.h5')
model <- load_model_hdf5('model_e500_b1.h5')

pred.nn <- model %>% predict(train_norm)

cat('RMSE train: ', rmse(dataset$y-pred.nn),
    '\nSD target: ', sd(train$y))

dataset.train = data.frame(x=dataset$x, pred = pred.nn)

g<- ggplot(dataset) + geom_point(aes(x,y), size = 0.5) +
  geom_line(data = datos_reales, aes(x,y), color = 'blue', size = 1) +
  geom_line(data = dataset.train, aes(x, pred), col = 'red', size = 2)
if( (n.anomalias !=0) & (tamano.anomalia !=0) ) {
  g + geom_point(data = dataset[fil.anomalias,], aes(x,y), col = 'blue', size = 2)
} else {
  g
}






