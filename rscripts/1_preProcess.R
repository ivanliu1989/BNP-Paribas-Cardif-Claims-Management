library(data.table)
setwd("C:/Users/iliu2/Documents/Kaggle_BNP/")
rm(list = ls()); gc(reset = TRUE)

train <- fread("./data/train.csv", stringsAsFactors = F, data.table = F, na.strings = "")
test <- fread("./data/test.csv", stringsAsFactors = F, data.table = F, na.strings = "")
par(mfcol = c(1,1))

#---------------
# Imputation ---
#---------------
test$target <- -1
all <- rbind(train, test)
all_naCol <- sapply(names(all[,-c(1,2)]), function(x) mean(is.na(all[,x])))

# train_naCol <- sapply(names(train[,-c(1,2)]), function(x) mean(is.na(train[,x])))
# test_naCol <- sapply(names(train[,-c(1,2)]), function(x) mean(is.na(train[,x])))
# plot(train_naCol, col = 'blue')
# points(test_naCol, col = 'red')

str(all[,names(all_naCol[all_naCol>0])], list.len = ncol(all))
str(all[,names(all_naCol[all_naCol==0])], list.len = ncol(all))
ordi <- c('v38', 'v62', 'v72', 'v129')
cate <- c('v3', 'v22', 'v24', 'v30', 'v31', 'v47', 'v52','v56', 'v66', 'v71', 'v74', 'v75', 'v79', 'v91', 'v107', 'v110', 'v112', 'v113', 'v125')
nume <- names(all[, !names(all) %in% c('ID', 'target', ordi, cate)])

par(mfcol = c(2,2))
# Ordinal
sapply(c('v38', 'v62', 'v72', 'v129'), function(x) plot(table(train[,x]), type = 'h', col = 'blue')) 
# Categorical 
sapply(c('v3', 'v22', 'v24', 'v30', 'v31', 'v47', 'v52','v56', 'v66', 'v71', 'v74', 'v75', 'v79', 'v91',
         'v107', 'v110', 'v112', 'v113', 'v125'), function(x) table(train[,x]))
# numerical
summary(all[, nume])

# Counts of NA
Cnt_NA_row <- apply(all, 1, function(x) sum(is.na(x)))

# Imputation
apply(all[, ordi], 2, function(x) mean(is.na(x))); str(all[,ordi])
all[, ordi][is.na(all[,ordi])] <- -1

apply(all[, cate], 2, function(x) mean(is.na(x))); str(all[,cate])
all[, cate][is.na(all[,cate])] <- '_NA'

apply(all[, nume], 2, function(x) mean(is.na(x))); str(all[,nume])
all[, nume][is.na(all[,nume])] <- -1

apply(all, 2, function(x) mean(is.na(x)))

#----------------------------------------
# Features creation / transformations ---
#----------------------------------------
# 1. Counts of NA, 0's, max, min, mean, sd
all$Cnt_NA_row <- Cnt_NA_row

# 2. Categorical variables: v91 - v107
table(all$v91); table(all$v107)
all$v91107 <- paste0(all$v91, all$v107)
all$v91107[which(all$v91107 == '_NA_NA')] <- '_NA';table(all$v91107)
all$v91 <- NULL; all$v107 <- NULL

# 3. Categorical variables: v71 - v75
# table(all[all$target == 1, 'v71']); table(all$v75)
# all$v7175 <- paste0(all$v71, all$v75)
# table(all$v7175)

# 4. Categorical variables: v79 - v71
# table(all$v79); table(all$v71)

# 5. Categorical variables: v10 - v31
# table(all$v10); table(all$v31)

# 6. Continuous variables

# 7. Hierarchy detect
#v56, v113, v125, v22

# 8. Bayesian: Encode categorical variables with its ratio of the target variable in train set.

# 9. Benford's Law / Log transformation

# 10. One-hot encoding
cate <- c('v3', 'v22', 'v24', 'v30', 'v31', 'v47', 'v52','v56', 'v66', 'v71', 'v74', 'v75', 'v79', 'v91107', 'v110', 'v112', 'v113', 'v125')
sapply(cate, function(x) length(table(all[,x])))
dummies <- dummyVars(Response ~ ., data = train[,c(1:127,ncol(train))], sep = "_", levelsOnly = FALSE, fullRank = TRUE)
train1 <- as.data.frame(predict(dummies, newdata = train[,c(1:127,ncol(train))]))
test_dum <- as.data.frame(predict(dummies, newdata = test[,c(1:127,ncol(train))]))
train_dum <- cbind(train1, Response=train$Response)
# head(train_dum[,names(table(names(train_dum))[table(names(train_dum))==2])])
zv <- names(table(names(train_dum))[table(names(train_dum))==2])
train_dum <- train_dum[,-which(names(train_dum) %in% zv)]
test_dum <- test_dum[,-which(names(test_dum) %in% zv)]

# 11. Dist
library(caret)
centroids <- classDist(all[, feature.names], as.factor(total_new[, 'Response']), pca = T, keep = 275) 
distances <- predict(centroids, total_new[, feature.names])
distances <- as.data.frame(distances)
distances_all <- distances[,-1]; names(distances_all) <- paste('DistALL', 1:8, sep = "")

# 12. tsne/kmeans
library(Rtsne)
tsne <- Rtsne(as.matrix(total_new[,feature.names]), dims = 3, perplexity=30, check_duplicates = F, pca = F, theta=0.5) #max_iter = 300, 
embedding <- as.data.frame(tsne$Y)
tsne_all <- embedding[,1:3]; names(tsne_all) <- c('TSNE_A1','TSNE_A2','TSNE_A3')

# 13. Genetic programming to automatically create non-linear features

# 14. Recursive Feature Elimination

# 15. Automation

#-----------------
# Adding noise ---
#-----------------


#----------
# Split ---
#----------
