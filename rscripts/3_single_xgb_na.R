setwd('/Users/ivanliu/Downloads/Kaggle_BNP')
library(xgboost)
library(caret)
rm(list=ls());gc()
load('./data/train_test_20160305.RData')
load('./BNP-Paribas-Cardif-Claims-Management/meta data/na_meta_data_20160307.RData')
ls()
### Split Data ###
set.seed(23)
cv <- 10
folds <- createFolds(train$target, k = cv, list = FALSE)
dropitems <- c('ID','target','TSNE_A1','TSNE_A2','TSNE_A3', 'DistALL1', 'DistALL2')
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

na_feat <- data.frame(km_cluster_all=km_cluster_all, km_cluster_tsne=km_cluster_tsne, tsne_na)
train_na_feat <- all[1:nrow(train),]
### Start Training ###
for(i in 1:cv){
    f <- folds==i
    # km_cluster_all, km_cluster_tsne, tsne_na
    dval          <- xgb.DMatrix(data=data.matrix(train_na_feat[f,]),label=train[f,'target'])
    dtrain        <- xgb.DMatrix(data=data.matrix(train_na_feat[!f,]),label=train[!f,'target']) 
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
}

all <- rbind(train,test)
all <- cbind(all, tsne_na)
train <- all[all$target>=0,]
test <- all[all$target<0,]
save(train, test, file = './data/train_test_20160308.RData')
