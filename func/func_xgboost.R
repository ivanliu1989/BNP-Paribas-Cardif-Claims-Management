gc()

doXGB <- function(train=train, preproc = FALSE, cv = 10){
            library(xgboost)
            library(caret)
            
            ### Split Data ###
            set.seed(23)
            folds <- createFolds(train$target, k = cv, list = FALSE)
            dropitems <- c('ID','target')
            feature.names <- names(train)[!names(train) %in% dropitems] 
            
            if(preproc){
                sc <- preProcess(train[,feature.names],method = c('center', 'scale'))
                train <- cbind(ID = train$ID, predict(sc, train[,feature.names]), target = train$target)
            }
        
            ### Setup Results Table ###
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
                
                if(i == 1){
                    cv_score = clf$bestScore
                }else{
                    cv_score = cv_score + clf$bestScore
                }
                ### Make predictions
                cat(paste0('Iteration: ', i, ' || Score: ', clf$bestScore))
            }
            return(cv_score/cv)
}


az_to_int <- function(az) {
    xx <- strsplit(tolower(az), "")[[1]]
    pos <- match(xx, letters[(1:26)]) 
    result <- sum( pos* 26^rev(seq_along(xx)-1))
    return(result)
}