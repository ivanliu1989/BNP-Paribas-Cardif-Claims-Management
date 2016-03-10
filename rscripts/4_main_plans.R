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
    
    # cv_score <- doXGB(train = all[all$target >= 0,], preproc = FALSE, cv = 5)
    # plot(all[all$target >= 0, c('target','Cnt_0_row')])
    # 0.4617784

# 2.    Imputation    
    all_naCol <- sapply(names(all[,-c(1,2)]), function(x) mean(is.na(all[,x])))
    str(all[,-c(1,2)],list.len = 500)
    all[, -c(1,2)][is.na(all[,-c(1,2)])] <- -999
    apply(all, 2, function(x) mean(is.na(x)))
    
    # cv_score <- doXGB(train = all[all$target >= 0,], preproc = FALSE, cv = 5)
    # 
    
# 2.5    Categorical variables: v91 - v107
    table(all$v91); table(all$v107)
    all$v91107 <- paste0(all$v91, all$v107)
    all$v91107[which(all$v91107 == '-999-999')] <- '-999';table(all$v91107)
    all$v91 <- NULL; all$v107 <- NULL
    all$v91107 <- as.numeric(all$v91107) 
    
    cv_score <- doXGB(train = all[all$target >= 0,], preproc = FALSE, cv = 5)
    #
    
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
    
    cv_score <- doXGB(train = all[all$target >= 0,], preproc = FALSE, cv = 5)
    #
    
# 4.    Dist/tSNE

# 5.    Bayesian enconding

# 6.    Remove small categories
# v22, v56
# v47, v71, v79, v113
dim(table(train[-which(train$v22 %in% unique(test$v22)),'v22']))
dim(table(test[-which(test$v22 %in% unique(train$v22)),'v22']))

table(train[-which(train$v56 %in% unique(test$v56)),'v56'])
table(test[-which(test$v56 %in% unique(train$v56)),'v56'])

# 7.    Find high correlations

# 8.    Zero variance

# 9.    Kmeans separation to find NA patterns

# 10.   One hot encoding

# 11.   Scale

# 12.   Imputation for non-systematic variables

# 13.   Feature importance detection










################
# MODELS: ######
################
# 1.    Vowpal Wabbit: two way or three way interactions

# 2.    Xgboost

# 3.    H2O deep learning and logistic regression

# 4.    LIBFM

# 5.    LIBSVM