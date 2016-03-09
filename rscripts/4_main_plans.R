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
    for(f in cate){
        all[,f]<-sapply(all[,f], az_to_int)
    }
    head(all[,cate])
    
    cv_score <- doXGB(train = all[all$target >= 0,], preproc = FALSE, cv = 5)

# 1.    Bayesian enconding

# 2.    Dist/tSNE

# 3.    0s counts

# 4.    Test effectiveness of hierarchies

# 5.    Remove small categories
    # v22, v56
    # v47, v71, v79, v113
    dim(table(train[-which(train$v22 %in% unique(test$v22)),'v22']))
    dim(table(test[-which(test$v22 %in% unique(train$v22)),'v22']))
    
    table(train[-which(train$v56 %in% unique(test$v56)),'v56'])
    table(test[-which(test$v56 %in% unique(train$v56)),'v56'])

# 6.    Find high correlations

# 7.    Zero variance

# 8.    Kmeans separation to find NA patterns

# 9.    One hot encoding

# 10.   Scale

# 11.   Imputation for non-systematic variables

# 12.   Feature importance detection

    
    
    
    
    
    
    
    
    
################
# MODELS: ######
################
# 1.    Vowpal Wabbit: two way or three way interactions

# 2.    Xgboost

# 3.    H2O deep learning and logistic regression

# 4.    LIBFM

# 5.    LIBSVM