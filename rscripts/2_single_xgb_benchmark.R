setwd('/Users/ivanliu/Downloads/Kaggle_BNP')
library(xgboost)
library(caret)
rm(list=ls());gc()
load('./data/train_test_20160305.RData')

### Split Data ###
set.seed(23)
cv <- 10
folds <- createFolds(train$target, k = cv, list = FALSE)
dropitems <- c('ID','target')
feature.names <- names(train)[!names(train) %in% dropitems] 
# sc <- preProcess(train[,feature.names],method = c('center', 'scale'))
# train <- cbind(ID = train$ID, predict(sc, train[,feature.names]), target = train$target)

### Setup Results Table ###
results <- as.data.frame(matrix(rep(0,3*cv), cv))
names(results) <- c('cv_num', 'AUC', 'LogLoss')

nr <- 1500
sr <- 300
sp <- 0.01
md <- 11
mcw <- 1
ss <- 0.96
cs <- 0.45

### Start Training ###
for(i in 1:cv){
    f <- folds==i
    dval          <- xgb.DMatrix(data=data.matrix(train[f,feature.names]),label=train[f,'target'])
    dtrain        <- xgb.DMatrix(data=data.matrix(train[!f,feature.names]),label=train[!f,'target']) 
    watchlist     <- list(val=dval,train=dtrain)
    
    clf <- xgb.train(data                = dtrain,
                     nrounds             = nr, 
                     early.stop.round    = sr,
                     watchlist           = watchlist,
                     eval_metric         = 'logloss',
                     maximize            = FALSE,
                     objective           = "binary:logistic",
                     booster             = "gbtree",
                     eta                 = sp,
                     max_depth           = md,
                     min_child_weight    = mcw,
                     subsample           = ss,
                     colsample           = cs,
                     print.every.n       = 200
    )
    
    ### Make predictions
    cat(paste0('Iteration: ', i, ' || Score: ', clf$bestScore))
    # 0.463547
    # 0.456657
    # 0.457595
    # 0.454999
    # 0.459592
    
    # 0.465319
    # 0.457349
    # 0.458969
    # 0.456499
    # 0.461414
}

# For test data
dtest          <- xgb.DMatrix(data=data.matrix(test[,feature.names]),label=test[,'target'])
dtrain        <- xgb.DMatrix(data=data.matrix(train[,feature.names]),label=train[,'target']) 
watchlist     <- list(val=dtrain,train=dtrain)

clf <- xgb.train(data                = dtrain,
                 nrounds             = nr, 
                 early.stop.round    = sr,
                 watchlist           = watchlist,
                 eval_metric         = 'logloss',
                 maximize            = FALSE,
                 objective           = "binary:logistic",
                 booster             = "gbtree",
                 eta                 = sp,
                 max_depth           = md,
                 min_child_weight    = mcw,
                 subsample           = ss,
                 colsample           = cs,
                 print.every.n       = 10
)

### Make predictions
validPreds <- predict(clf, dtest)
prediction <- cbind(ID = test[,'ID'], target = validPreds)
write(prediction, file = './submissions/single_xgb_20150305_1.csv')