# Pruebas sobre House Price

# Regresion Lineal

rmse <- function(error) sqrt(mean(error^2, na.rm = TRUE))

data <- train1
data$SalePrice <- all$SalePrice[!is.na(all$SalePrice)]

lm.model <- lm(SalePrice ~ ., data = data)

pred.lm <- predict(lm.model, data=train1)

summary(lm.model)
rmse(data$SalePrice - pred.lm)

predictions_lm <- exp(predict(lm.model, newdata=test1))

sub_lm <- data.frame(Id = test_labels, SalePrice = predictions_lm)
head(sub_lm)
write.csv(sub_lm, file = 'sub_lm.csv', row.names = F)
# 0.12717 frente al xgboost+lasso 0.11979
# Se traduce en:
exp(0.12717)
exp(0.11979)

# Y cuánto es el RMSE de la media = desviación estándar
sd(data$SalePrice, na.rm=TRUE)
exp(0.3997154)


sub_avg3 <- data.frame(Id = test_labels, SalePrice = (predictions_XGB+2*predictions_lasso+predictions_lm)/4)
head(sub_avg3)
write.csv(sub_avg3, file = 'average3.csv', row.names = F)


# Prueba con lightgbm sin preprocesado

dataset <- train %>% bind_rows(test)

dataset <- dataset %>% mutate_if(is.character, function(x) as.numeric(as.factor(x)))

train_2 <- dataset[1:nrow(train),]
test_2 <- dataset[(nrow(train)+1):nrow(dataset),]

# dtrain_2 <- xgb.DMatrix(data = as.matrix(train_2 %>% select(-SalePrice)), label= train_2$SalePrice)
# dtest_2 <- xgb.DMatrix(data = as.matrix(test_2 %>% select(-SalePrice)))

dtrain_2 <- lgb.Dataset(data = as.matrix(train_2 %>% select(-SalePrice)), label= as.matrix(train_2$SalePrice))

default_param_2<-list(
  objective = "regression",
  metric = 'rmse',
  learning_rate=0.05,
  num_leaves=20,
  bagging_fraction=1,
  feature_fraction=0.8
)

lgb_mod_2 <- lgb.train(data = dtrain_2, param=default_param_2, verbose = 1, eval_freq = 50, nrounds = 400)

rmse(log(train_2$SalePrice) - log(predict(lgb_mod_2, as.matrix(train_2 %>% select(-SalePrice)))) )

predictions_LGB_2 <- predict(lgb_mod_2, as.matrix(test_2 %>% select(-SalePrice)))

sub_lgb_2 <- data.frame(Id = test_labels, SalePrice = predictions_LGB_2)
head(sub_lgb_2)
write.csv(sub_lgb_2, file = 'Sub_lgb_2.csv', row.names = F)
# LB: 0.13210
exp(0.13210)
