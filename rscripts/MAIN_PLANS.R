#############################
# FEATURE ENGINEERING: ######
#############################
# 1.       Bayesian enconding

# 2.       Dist/tSNE

# 3.       0s counts

# 4.       Test effectiveness of hierarchies

# 5.       Remove small categories
    # v22, v56
    # v47, v71, v79, v113
    dim(table(train[-which(train$v22 %in% unique(test$v22)),'v22']))
    dim(table(test[-which(test$v22 %in% unique(train$v22)),'v22']))
    
    table(train[-which(train$v56 %in% unique(test$v56)),'v56'])
    table(test[-which(test$v56 %in% unique(train$v56)),'v56'])

# 6.       Find high correlations

# 7.       Zero variance

# 8.       Kmeans separation to find NA patterns

# 9.       One hot encoding

# 10.   Scale

# 11.   Imputation for non-systematic variables

# 12.   Feature importance detection

################
# MODELS: ######
################
# 1.       Vowpal Wabbit: two way or three way interactions

# 2.       Xgboost

# 3.       H2O deep learning and logistic regression

# 4.       LIBFM

# 5.       LIBSVM