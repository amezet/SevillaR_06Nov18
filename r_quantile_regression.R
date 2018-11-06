
##########################
#
#   QUANTILE REGRESSION
#
###########################

#ADAPTATION OF PYTHON EXAMPLE
#https://towardsdatascience.com/deep-quantile-regression-c85481548b5a

#################
# INSTALL KERAS
#################

#devtools::install_github("rstudio/tensorflow") 
#devtools::install_github("rstudio/keras") 

#WE CREATE CONDA ENVIROMENT WITH TF AND KERAS:
#library(keras)
#install_keras()

#################
# LIBRARIES
#################

library(keras)
library(ggplot2)
library(caret)

##################
#   CREATE DATA
##################

#NUMBER OF ROWS
n = 1000
#360 DAYS * 3 YEARS
t = runif(n, min = -2, max = 2)
#SIN WITH NOISE
y = (t)^2+rnorm(n)*0.3*(t+2.5)
#CREATE DF
df = data.frame(t,y)
plot(df)

###################
#   TO ARRAY
###################

X_train = array(df$t)
Y_train = array(df$y)

###################
#
#   KERAS
#
###################

#NN ARCHITECTURE
model = keras_model_sequential()
model %>% 
  layer_dense(units = 20, input_shape = 1, activation = 'relu')%>%
  layer_dense(units = 20, activation = 'relu')%>%
  layer_dense(units=1)

#CHOOSE ERROR
model%>% compile(loss = 'mae',metrics = 'mae', optimizer = 'adam')

#FIT
model%>% fit(X_train,Y_train,epochs=100, batch_size=32)

#PRED
df$pred = c(model%>%predict(X_train))

#############
#   PLOT
#############

ggplot(df, aes(t, y)) +
  geom_point() +
  theme_minimal()+
  geom_point(aes(t,pred),col = 'red')

###################################
#
#   QUANTILE LOSS WITH KERAS
#
###################################

#NN ARCHITECTURE
k_clear_session()
model_q = keras_model_sequential()
model_q %>% 
  layer_dense(units = 20, input_shape = 1, activation = 'relu')%>%
  layer_dense(units = 20, activation = 'relu')%>%
  layer_dense(units=1)

#CHOOSE ERROR
quantile <- 0.9
tilted_loss <- function(q, y, f) {
  e <- y - f
  k_mean(k_maximum(q * e, (q - 1) * e), axis = 2)
}

model_q %>% compile(
  optimizer = "adam",
  loss = function(y_true, y_pred)
    tilted_loss(quantile, y_true, y_pred),
  metrics = "mae"
)

#FIT
model_q%>% fit(X_train,Y_train,epochs=100, batch_size=32)

#PREDICT
df$pred_q = c(model_q%>%predict(X_train))

##############
#   PLOT
##############

ggplot(df, aes(t, y)) +
  geom_point() +
  theme_minimal()+
  geom_point(aes(t,pred),col = 'red')+
  geom_point(aes(t,pred_q),col = 'blue')


