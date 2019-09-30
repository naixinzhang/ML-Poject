rm(list=ls())
setwd("D:/Documents/Wisconsin/AAE722")
dat <-
  read.csv(file = "ccdefault_data.csv", header = T, sep = ",")
library(dplyr)
library(caret)
library(pROC)
library(ggplot2)
library(plotROC)
names(dat)[names(dat)=="PAY_0"] <- "PAY_1"
dat <- dat[,-1]
pay_var <- paste0("PAY_", 1:6)
factor_var <- c("SEX", "EDUCATION", "MARRIAGE", pay_var, "default.payment.next.month")
dat[factor_var] <- lapply(dat[factor_var], factor)
library(corrgram)
corrgram(dat, order = TRUE, lower.panel = panel.shade,
         upper.panel = panel.pie, text.panel = panel.txt)
corrgram(dat, order = TRUE, lower.panel = panel.ellipse,
         upper.panel = panel.pts, text.panel = panel.txt,
         diag.panel = panel.minmax)
levels(dat$MARRIAGE)[levels(dat$MARRIAGE) == "1"] <- "Married"
levels(dat$MARRIAGE)[levels(dat$MARRIAGE) == "2"] <- "Single"
levels(dat$MARRIAGE)[levels(dat$MARRIAGE) == "3"] <- "Other"
set.seed(123)
numrow <- nrow(dat)
train_ind <- sample(1:numrow,size = as.integer(0.7*numrow))
dat_train = dat[train_ind,]
dat_test = dat[-train_ind,]
nzv <- nearZeroVar(dat,
                   freqCut = 95/5,
                   uniqueCut = 10,
                   saveMetrics= TRUE,
                   foreach = FALSE,
                   allowParallel = TRUE)
nzv
for (x in factor_var) {
  levels(dat_train[, x]) <- make.names(levels(dat_train[, x]))
  levels(dat_test[, x])  <- make.names(levels(dat_test[, x]))
}
dat_train[pay_var] <- lapply(dat_train[pay_var], relevel, ref = "X.1")
dat_test[pay_var]  <- lapply(dat_test[pay_var], relevel, ref = "X.1")
dat_train["EDUCATION"] <- lapply(dat_train["EDUCATION"], relevel, ref = "X1")
dat_test["EDUCATION"]  <- lapply(dat_test["EDUCATION"], relevel, ref = "X1")
dat_train["MARRIAGE"] <- lapply(dat_train["MARRIAGE"], relevel, ref = "Married")
dat_test["MARRIAGE"]  <- lapply(dat_test["MARRIAGE"], relevel, ref = "Married")
library(doParallel)
cl <- makePSOCKcluster(5)
registerDoParallel(cl)
control <- trainControl(method = "cv",
                        number = 10,
                        summaryFunction = twoClassSummary,
                        classProbs = TRUE)
model_glm <- train(default.payment.next.month ~ .,
                   data = dat_train,
                   method = "glm",
                   family = "binomial",
                   trControl = control,
                   metric = "ROC")
model_knn <- train(default.payment.next.month ~ .,
                   data = dat_train,
                   method = "knn",
                   tuneGrid = expand.grid(k = 1:10),
                   trControl = control,
                   metric = "ROC")
model_ridge <- train(default.payment.next.month ~ .,
                     data = dat_train,
                     method = "glmnet",
                     tuneGrid = expand.grid(alpha = 0,
                                            lambda = c(0.001, 0.002, 0.005,
                                                       0.01, 0.02, 0.05,
                                                       0.1, 0.2, 0.5, 1, 2, 5)),
                     trControl = control,
                     metric = "ROC")
model_lasso <- train(default.payment.next.month ~ .,
                     data = dat_train,
                     method = "glmnet",
                     tuneGrid = expand.grid(alpha = 1,
                                            lambda = c(0.001, 0.002, 0.005,
                                                       0.01, 0.02, 0.05,
                                                       0.1, 0.2, 0.5, 1, 2, 5)),
                     trControl = control,
                     metric = "ROC")
model_lasso$finalModel
model_cart <- train(default.payment.next.month ~ .,
                    data = dat_train,
                    method = "rpart",
                    trControl = control,
                    metric = "ROC")
model_lasso2 <- glmnet(default.payment.next.month ~ ., data = dat_train, lambda = 0.002)
model_rf <- train(default.payment.next.month ~ .,
                  data = dat_train,
                  method = "ranger",
                  trControl = control,
                  metric = "ROC")
ModelInterpret <- function(model, train, test){
  pred <- predict(model, newdata = test %>% select(-default.payment.next.month), type = "raw")
  summary(pred)
  confusionMatrix(pred, test$default.payment.next.month)
}
Defaultrate <- function(model, train, test){
  pred <- predict(model, newdata = test %>% select(-default.payment.next.month), type = "raw")
  default_rate <- as.numeric(summary(pred)["X1"]/nrow(dat_test))
  print("The default rate next month is")
  default_rate
}
ModelInterpret(model_glm, dat_train, dat_test)
Defaultrate(model_glm, dat_train, dat_test)
model_glm$bestTune
ModelInterpret(model_knn, dat_train, dat_test)
Defaultrate(model_knn, dat_train, dat_test)
model_knn$bestTune
ModelInterpret(model_ridge, dat_train, dat_test)
Defaultrate(model_ridge, dat_train, dat_test)
model_ridge$bestTune
ModelInterpret(model_lasso, dat_train, dat_test)
Defaultrate(model_lasso, dat_train, dat_test)
model_lasso$bestTune
ModelInterpret(model_cart, dat_train, dat_test)
Defaultrate(model_cart, dat_train, dat_test)
model_cart$bestTune
ModelInterpret(model_rf, dat_train, dat_test)
Defaultrate(model_rf, dat_train, dat_test)
model_rf$bestTune
library(rpart.plot)
rpart.plot(model_cart$finalModel)
model_logit <- glm(default.payment.next.month ~ .,
                   data = dat,
                   family = "binomial")
summary(model_logit)
library(dagitty)
library(lavaan)
g <- dagitty('dag {
              LIMIT_BAL [pos = "0,1"]
              MARRIAGE [pos = "1,0"]
              AGE [pos = "0,0"]
              PAY_AMT [pos = "1,2"]
              BILL_AMT [pos = "0,2"]
              PAY_1 [pos = "2,1"]
              PAY_X [pos = "2,0"]
              default [pos = "1,1"]
              
              AGE -> LIMIT_BAL -> default
              AGE -> MARRIAGE -> default
              BILL_AMT -> PAY_AMT -> default
              PAY_X -> PAY_1 -> default
              }')
plot(g)