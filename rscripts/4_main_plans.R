setwd('/Users/ivanliu/Downloads/Kaggle_BNP')
library(data.table)
rm(list=ls());gc()
source('./BNP-Paribas-Cardif-Claims-Management/func/func_xgboost.R')

train <- fread("./data/train.csv", stringsAsFactors = F, data.table = F, na.strings = "")
test <- fread("./data/test.csv", stringsAsFactors = F, data.table = F, na.strings = "")

#############################
# FEATURE ENGINEERING: ######
#############################
test$target <- -1
all <- rbind(train, test)
all_raw <- rbind(train, test)
ordi <- c('v38', 'v62', 'v72', 'v129')
cate <- c('v3', 'v22', 'v24', 'v30', 'v31', 'v47', 'v52','v56', 'v66', 'v71', 'v74', 'v75', 'v79', 'v91', 'v107', 'v110', 'v112', 'v113', 'v125')
nume <- names(all[, !names(all) %in% c('ID', 'target', ordi, cate)])

# 0.    Factorize
    for (f in cate){
        all[,f]<-sapply(all[,f], az_to_int)
    }
    head(all[,cate])
    
    # cv_score <- doXGB(train = all[all$target >= 0,], preproc = FALSE, cv = 5)
    # 0.4619718

# 1.    0s counts
    N <- ncol(all)-2
    Cnt_NA_row <- apply(all[,-c(1,2)], 1, function(x) sum(is.na(x))/N)
    Cnt_0_row <- apply(all[,-c(1,2)], 1, function(x) sum(x==0, na.rm = T)/N)
    all$Cnt_NA_row <- Cnt_NA_row
    all$Cnt_0_row <- Cnt_0_row
    
    # cv_score_1 <- doXGB(train = all[all$target >= 0,], preproc = FALSE, cv = 5)
    # plot(all[all$target >= 0, c('target','Cnt_0_row')])
    # 0.4617784

# 2.    Imputation    
    all_naCol <- sapply(names(all[,-c(1,2)]), function(x) mean(is.na(all[,x])))
    str(all[,-c(1,2)],list.len = 500)
    all[, -c(1,2)][is.na(all[,-c(1,2)])] <- -999
    apply(all, 2, function(x) mean(is.na(x)))
    
    # cv_score_2 <- doXGB(train = all[all$target >= 0,], preproc = FALSE, cv = 5)
    # 0.4622458
    
# 2.5    Categorical variables: v91 - v107
    table(all$v91); table(all$v107)
    all$v91107 <- paste0(all$v91, all$v107)
    all$v91107[which(all$v91107 == '-999-999')] <- '-999';table(all$v91107)
    all$v91 <- NULL; all$v107 <- NULL
    all$v91107 <- as.numeric(all$v91107) 
    
    # cv_score_2.5 <- doXGB(train = all[all$target >= 0,], preproc = FALSE, cv = 5)
    # 0.4623838
    
# 3.    Test effectiveness of hierarchies
    # v56, v113, v125, v22
    library(stringr)
    #v22 - Bayesian
    v22 <- all_raw$v22; v22[v22 == '_NA'] <- ""
    v22 <- as.data.frame(str_split_fixed(v22, "", 4)); names(v22) <- paste0('v22_', 1:4)
    for(x in 1:4){v22[,x] <-  as.character(v22[,x])};str(v22);v22[v22 == ""] <- NA
    for(x in 1:4){v22[,x] <-  sapply(v22[,x], az_to_int)};v22[is.na(v22)] <- -999
    apply(v22, 2, table); head(v22)
    
    #v56
    v56 <- all_raw$v56; v56[v56 == '_NA'] <- ""
    v56 <- as.data.frame(str_split_fixed(v56, "", 2)); names(v56) <- paste0('v56_', 1:2)
    for(x in 1:2){v56[,x] <-  as.character(v56[,x])};str(v56);v56[v56 == ""] <- NA
    for(x in 1:2){v56[,x] <-  sapply(v56[,x], az_to_int)};v56[is.na(v56)] <- -999
    apply(v56, 2, table); head(v56)
    
    #v113
    v113 <- all_raw$v113; v113[v113 == '_NA'] <- ""
    v113 <- as.data.frame(str_split_fixed(v113, "", 2)); names(v113) <- paste0('v113_', 1:2)
    for(x in 1:2){v113[,x] <-  as.character(v113[,x])};str(v113);v113[v113 == ""] <- NA
    for(x in 1:2){v113[,x] <-  sapply(v113[,x], az_to_int)};v113[is.na(v113)] <- -999
    apply(v113, 2, table); head(v113)
    
    #v125 - Bayesian
    v125 <- all_raw$v125; v125[v125 == '_NA'] <- ""
    v125 <- as.data.frame(str_split_fixed(v125, "", 2)); names(v125) <- paste0('v125_', 1:2)
    for(x in 1:2){v125[,x] <-  as.character(v125[,x])};str(v125);v125[v125 == ""] <- NA
    for(x in 1:2){v125[,x] <-  sapply(v125[,x], az_to_int)};v125[is.na(v125)] <- -999
    apply(v125, 2, table); head(v125)
    
    all <- cbind(all, v22,v56,v113,v125)
    
    # cv_score_3 <- doXGB(train = all[all$target >= 0,], preproc = FALSE, cv = 5)
    # 0.4616048
    
# 4.    Dist/tSNE
    load('./BNP-Paribas-Cardif-Claims-Management/meta data/meta_data_20160305.RData')
    all <- cbind(all, distances_all, tsne_all)
    
    # cv_score_4 <- doXGB(train = all[all$target >= 0,], preproc = FALSE, cv = 5)
    # 0.4599826
    
# 5.    Bayesian encoding
    all_bayesian <- all[all$target >= 0, ]
    dim(train); dim(all_bayesian)
    
    v24_bayes <- sapply(names(table(all_bayesian$v24)), function(x) mean(all_bayesian[all_bayesian$v24 == x, 'target']))
    v30_bayes <- sapply(names(table(all_bayesian$v30)), function(x) mean(all_bayesian[all_bayesian$v30 == x, 'target']))
    v31_bayes <- sapply(names(table(all_bayesian$v31)), function(x) mean(all_bayesian[all_bayesian$v31 == x, 'target']))
    v52_bayes <- sapply(names(table(all_bayesian$v52)), function(x) mean(all_bayesian[all_bayesian$v52 == x, 'target']))
    v66_bayes <- sapply(names(table(all_bayesian$v66)), function(x) mean(all_bayesian[all_bayesian$v66 == x, 'target']))
    v91107_bayes <- sapply(names(table(all_bayesian$v91107)), function(x) mean(all_bayesian[all_bayesian$v91107 == x, 'target']))
    v110_bayes <- sapply(names(table(all_bayesian$v110)), function(x) mean(all_bayesian[all_bayesian$v110 == x, 'target']))
    v112_bayes <- sapply(names(table(all_bayesian$v112)), function(x) mean(all_bayesian[all_bayesian$v112 == x, 'target']))
    v125_bayes <- sapply(names(table(all_bayesian$v125)), function(x) mean(all_bayesian[all_bayesian$v125 == x, 'target']))
    v22_1_bayes <- sapply(names(table(all_bayesian$v22_1)), function(x) mean(all_bayesian[all_bayesian$v22_1 == x, 'target']))
    v22_2_bayes <- sapply(names(table(all_bayesian$v22_2)), function(x) mean(all_bayesian[all_bayesian$v22_2 == x, 'target']))
    v22_3_bayes <- sapply(names(table(all_bayesian$v22_3)), function(x) mean(all_bayesian[all_bayesian$v22_3 == x, 'target']))
    v22_4_bayes <- sapply(names(table(all_bayesian$v22_4)), function(x) mean(all_bayesian[all_bayesian$v22_4 == x, 'target']))
    v125_1_bayes <- sapply(names(table(all_bayesian$v125_1)), function(x) mean(all_bayesian[all_bayesian$v125_1 == x, 'target']))
    v125_2_bayes <- sapply(names(table(all_bayesian$v125_2)), function(x) mean(all_bayesian[all_bayesian$v125_2 == x, 'target']))
    
    all[,"v24_bayes"]<-data.frame(v24_bayes)[all[,"v24"], "v24_bayes"]
    for(i in 1:length(names(v30_bayes))){all[all$v30 == as.numeric(names(v30_bayes)[i]),"v30_bayes"] <- v30_bayes[[i]]}
    for(i in 1:length(names(v31_bayes))){all[all$v31 == as.numeric(names(v31_bayes)[i]),"v31_bayes"] <- v31_bayes[[i]]}
    for(i in 1:length(names(v52_bayes))){all[all$v52 == as.numeric(names(v52_bayes)[i]),"v52_bayes"] <- v52_bayes[[i]]}
    for(i in 1:length(names(v66_bayes))){all[all$v66 == as.numeric(names(v66_bayes)[i]),"v66_bayes"] <- v66_bayes[[i]]}
    for(i in 1:length(names(v91107_bayes))){all[all$v91107 == as.numeric(names(v91107_bayes)[i]),"v91107_bayes"] <- v91107_bayes[[i]]}
    for(i in 1:length(names(v110_bayes))){all[all$v110 == as.numeric(names(v110_bayes)[i]),"v110_bayes"] <- v110_bayes[[i]]}
    for(i in 1:length(names(v112_bayes))){all[all$v112 == as.numeric(names(v112_bayes)[i]),"v112_bayes"] <- v112_bayes[[i]]}
    for(i in 1:length(names(v125_bayes))){all[all$v125 == as.numeric(names(v125_bayes)[i]),"v125_bayes"] <- v125_bayes[[i]]}
    for(i in 1:length(names(v22_1_bayes))){all[all$v22_1 == as.numeric(names(v22_1_bayes)[i]),"v22_1_bayes"] <- v22_1_bayes[[i]]}
    for(i in 1:length(names(v22_2_bayes))){all[all$v22_2 == as.numeric(names(v22_2_bayes)[i]),"v22_2_bayes"] <- v22_2_bayes[[i]]}
    for(i in 1:length(names(v22_3_bayes))){all[all$v22_3 == as.numeric(names(v22_3_bayes)[i]),"v22_3_bayes"] <- v22_3_bayes[[i]]}
    for(i in 1:length(names(v22_4_bayes))){all[all$v22_4 == as.numeric(names(v22_4_bayes)[i]),"v22_4_bayes"] <- v22_4_bayes[[i]]}
    for(i in 1:length(names(v125_1_bayes))){all[all$v125_1 == as.numeric(names(v125_1_bayes)[i]),"v125_1_bayes"] <- v125_1_bayes[[i]]}
    for(i in 1:length(names(v125_2_bayes))){all[all$v125_2 == as.numeric(names(v125_2_bayes)[i]),"v125_2_bayes"] <- v125_2_bayes[[i]]}
    
    # cv_score_5 <- doXGB(train = all[all$target >= 0,], preproc = FALSE, cv = 5)
    # 0.4586156
    
# 6.    Remove small categories
    train <- all[all$target >= 0,]
    test <- all[all$target < 0,]
    dim(train); dim(test)
    # v22, v56
    # v47, v71, v79, v113
    # dim(table(train[-which(train$v22 %in% unique(test$v22)),'v22']))
    # dim(table(test[-which(test$v22 %in% unique(train$v22)),'v22']))
    
    v22_sm_cat <- as.numeric(c(names(table(train[-which(train$v22 %in% unique(test$v22)),'v22'])), names(table(test[-which(test$v22 %in% unique(train$v22)),'v22']))))
    v56_sm_cat <- as.numeric(c(names(table(train[-which(train$v56 %in% unique(test$v56)),'v56'])), names(table(test[-which(test$v56 %in% unique(train$v56)),'v56']))))
    v47_sm_cat <- as.numeric(c(names(table(train[-which(train$v47 %in% unique(test$v47)),'v47'])), names(table(test[-which(test$v47 %in% unique(train$v47)),'v47']))))
    v71_sm_cat <- as.numeric(c(names(table(train[-which(train$v71 %in% unique(test$v71)),'v71'])), names(table(test[-which(test$v71 %in% unique(train$v71)),'v71']))))
    v79_sm_cat <- as.numeric(c(names(table(train[-which(train$v79 %in% unique(test$v79)),'v79'])), names(table(test[-which(test$v79 %in% unique(train$v79)),'v79']))))
    v113_sm_cat <- as.numeric(c(names(table(train[-which(train$v113 %in% unique(test$v113)),'v113'])), names(table(test[-which(test$v113 %in% unique(train$v113)),'v113']))))
    
    all[all$v22 %in% v22_sm_cat, 'v22'] <- -999
    all[all$v56 %in% v56_sm_cat, 'v56'] <- -999
    all[all$v47 %in% v47_sm_cat, 'v47'] <- -999
    all[all$v71 %in% v71_sm_cat, 'v71'] <- -999
    all[all$v79 %in% v79_sm_cat, 'v79'] <- -999
    all[all$v113 %in% v113_sm_cat, 'v113'] <- -999

    # cv_score_6 <- doXGB(train = all[all$target >= 0,], preproc = FALSE, cv = 5)
    # 0.458649
    
# 7.    Find high correlations
    descrCor <- cor(all)
    summary(descrCor[upper.tri(descrCor)])
    
    highlyCorDescr <- findCorrelation(descrCor, cutoff = .99)
    filteredDescr <- all[,-highlyCorDescr]
    descrCor2 <- cor(filteredDescr)
    summary(descrCor2[upper.tri(descrCor2)])
    
    # cv_score_7 <- doXGB(train = filteredDescr[filteredDescr$target >= 0,], preproc = FALSE, cv = 5)
    # 0.459259 
    
# 8.    Zero variance
    nzv <- nearZeroVar(all, saveMetrics= TRUE)
    nzv[nzv$nzv,][1:10,]
    nzv <- nearZeroVar(all)
    filteredDescr <- all[, -nzv]
    dim(filteredDescr)
    
    # cv_score_8 <- doXGB(train = filteredDescr[filteredDescr$target >= 0,], preproc = FALSE, cv = 5)
    # 0.458745
    
# 9.    Kmeans separation to find NA patterns
    load('./BNP-Paribas-Cardif-Claims-Management/meta data/na_meta_data_20160307.RData')
    all <- cbind(all, km_cluster_all = km_cluster_all, km_cluster_tsne = km_cluster_tsne, tsne_na)
    
    # cv_score_9 <- doXGB(train = all[all$target >= 0,], preproc = FALSE, cv = 5)
    # 0.4587074
    
# 10.   One hot encoding
    cate <- c('v3', 'v24', 'v30', 'v31', 'v47', 'v52','v56', 'v66', 'v71', 'v74', 'v75', 'v79', 'v91107', 'v110', 'v112', 'v113', 'v125',
              'v22_1', 'v22_2', 'v22_3', 'v22_4', 'v56_1', 'v56_2', 'v113_1', 'v113_2', 'v125_1', 'v125_2', 'km_cluster_all', 'km_cluster_tsne')
    # v22
    sapply(cate, function(x) length(table(all[,x])))
    for(c in cate){all[,c] <- as.factor(all[,c])}; str(all)
    dummies <- dummyVars(target ~ ., data = all, sep = "_", levelsOnly = FALSE, fullRank = TRUE)
    all1 <- as.data.frame(predict(dummies, newdata = all))
    all1 <- cbind(all1, target = all$target)
    
    # cv_score_10 <- doXGB(train = all1[all1$target >= 0,], preproc = FALSE, cv = 5)
    # 0.458353
    
# 11.   Scale
    dropitems <- c('ID', 'target')
    feature.names <- names(all1)[!names(all1) %in% dropitems] 
    sc <- preProcess(all1[,feature.names],method = c('center', 'scale'))
    all1_sc <- cbind(ID = all1$ID, predict(sc, all1[,feature.names]), target = all1$target)

    # cv_score_11 <- doXGB(train = all1[all1$target >= 0,], preproc = FALSE, cv = 5)
    # 0.458353

# 12.   columns removal idea
    removeitems <- c('v8','v23','v25','v31','v36','v37','v46','v51','v53','v54','v63','v73','v75','v79','v81',
                   'v82','v89','v92','v95','v105','v108','v109','v110','v116','v117','v118','v119','v123','v124','v128') # ,'v107'
    head(all[,removeitems])
    all_rm <- all[,!names(all) %in% removeitems]
    
    # cv_score_12 <- doXGB(train = all_rm[all_rm$target >= 0,], preproc = FALSE, cv = 5)
    # 
        
# 13.   Benouilli Naive Bayes
    
# 14.   Imputation for non-systematic variables

# 15.   Feature selection mRMR











################
# MODELS: ######
################
# 1.    Vowpal Wabbit: two way or three way interactions

# 2.    Xgboost

# 3.    H2O deep learning and logistic regression

# 4.    LIBFM

# 5.    LIBSVM