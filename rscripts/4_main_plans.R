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
    
    cv_score <- doXGB(train = all[all$target >= 0,], preproc = FALSE, cv = 5)
    plot(all[all$target >= 0, c('target','Cnt_0_row')])
    # 

# 2.    Imputation    
    all_naCol <- sapply(names(all[,-c(1,2)]), function(x) mean(is.na(all[,x])))
    str(all[,-c(1,2)],list.len = 500)
    all[, -c(1,2)][is.na(all[,-c(1,2)])] <- -999
    apply(all, 2, function(x) mean(is.na(x)))
    
    cv_score <- doXGB(train = all[all$target >= 0,], preproc = FALSE, cv = 5)
    # 

# 3.    Bayesian enconding

# 4.    Dist/tSNE

# 5.    Test effectiveness of hierarchies

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

# 10.    One hot encoding

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