library(data.table)
library(moments)
# setwd("C:/Users/iliu2/Documents/Kaggle_BNP/")
setwd('/Users/ivanliu/Downloads/Kaggle_BNP')
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

# nume_na_mean <- colMeans(all[,nume], na.rm = T)
# ordi_na_mean <- colMeans(all[,ordi], na.rm = T)

# par(mfcol = c(2,2))
# # Ordinal
# sapply(c('v38', 'v62', 'v72', 'v129'), function(x) plot(table(train[,x]), type = 'h', col = 'blue')) 
# # Categorical 
# sapply(c('v3', 'v22', 'v24', 'v30', 'v31', 'v47', 'v52','v56', 'v66', 'v71', 'v74', 'v75', 'v79', 'v91',
#          'v107', 'v110', 'v112', 'v113', 'v125'), function(x) table(train[,x]))
# # numerical
# summary(all[, nume])

# Counts of NA
Cnt_NA_row <- apply(all, 1, function(x) sum(is.na(x)))

# Imputation
apply(all[, ordi], 2, function(x) mean(is.na(x))); str(all[,ordi])
all[, ordi][is.na(all[,ordi])] <- -999

apply(all[, cate], 2, function(x) mean(is.na(x))); str(all[,cate])
all[, cate][is.na(all[,cate])] <- '_NA'

apply(all[, nume], 2, function(x) mean(is.na(x))); str(all[,nume])
all[, nume][is.na(all[,nume])] <- -999

apply(all, 2, function(x) mean(is.na(x)))

#----------------------------------------
# Features creation / transformations ---
#----------------------------------------
# 1. Counts of NA, 0's, max, min, mean, sd
all$Cnt_NA_row <- Cnt_NA_row
# Cnt_sd_row <- apply(all[,nume], 1, function(x) mean(x > nume_na_mean))

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
# v56, v113, v125, v22
library(stringr)
#v22 - Bayesian
v22 <- all$v22; v22[v22 == '_NA'] <- ""
v22 <- as.data.frame(str_split_fixed(v22, "", 4)); names(v22) <- paste0('v22_', 1:4)
for(x in 1:4){v22[,x] <-  as.character(v22[,x])};str(v22);v22[v22 == ""] <- '_NA'
apply(v22, 2, table)

#v56
v56 <- all$v56; v56[v56 == '_NA'] <- ""
v56 <- as.data.frame(str_split_fixed(v56, "", 2)); names(v56) <- paste0('v56_', 1:2)
for(x in 1:2){v56[,x] <-  as.character(v56[,x])};str(v56);v56[v56 == ""] <- '_NA'
apply(v56, 2, table)

#v113
v113 <- all$v113; v113[v113 == '_NA'] <- ""
v113 <- as.data.frame(str_split_fixed(v113, "", 2)); names(v113) <- paste0('v113_', 1:2)
for(x in 1:2){v113[,x] <-  as.character(v113[,x])};str(v113);v113[v113 == ""] <- '_NA'
apply(v113, 2, table)

#v125 - Bayesian
v125 <- all$v125; v125[v125 == '_NA'] <- ""
v125 <- as.data.frame(str_split_fixed(v125, "", 2)); names(v125) <- paste0('v125_', 1:2)
for(x in 1:2){v125[,x] <-  as.character(v125[,x])};str(v125);v125[v125 == ""] <- '_NA'
apply(v125, 2, table)

all <- cbind(all, v22,v56,v113,v125)

# 8. Bayesian: Encode categorical variables with its ratio of the target variable in train set.
# v24, v30, v31, v52, v66, v91107, v110, v112, v125
# v22_1, v22_2, v22_3, v22_4, v125_1, v125_2
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
all[,"v30_bayes"]<-data.frame(v30_bayes)[all[,"v30"], "v30_bayes"]
all[,"v31_bayes"]<-data.frame(v31_bayes)[all[,"v31"], "v31_bayes"]
all[,"v52_bayes"]<-data.frame(v52_bayes)[all[,"v52"], "v52_bayes"]
all[,"v66_bayes"]<-data.frame(v66_bayes)[all[,"v66"], "v66_bayes"]
all[,"v91107_bayes"]<-data.frame(v91107_bayes)[all[,"v91107"], "v91107_bayes"]
all[,"v110_bayes"]<-data.frame(v110_bayes)[all[,"v110"], "v110_bayes"]
all[,"v112_bayes"]<-data.frame(v112_bayes)[all[,"v112"], "v112_bayes"]
all[,"v125_bayes"]<-data.frame(v125_bayes)[all[,"v125"], "v125_bayes"]
all[,"v22_1_bayes"]<-data.frame(v22_1_bayes)[all[,"v22_1"], "v22_1_bayes"]
all[,"v22_2_bayes"]<-data.frame(v22_2_bayes)[all[,"v22_2"], "v22_2_bayes"]
all[,"v22_3_bayes"]<-data.frame(v22_3_bayes)[all[,"v22_3"], "v22_3_bayes"]
all[,"v22_4_bayes"]<-data.frame(v22_4_bayes)[all[,"v22_4"], "v22_4_bayes"]
all[,"v125_1_bayes"]<-data.frame(v125_1_bayes)[all[,"v125_1"], "v125_1_bayes"]
all[,"v125_2_bayes"]<-data.frame(v125_2_bayes)[all[,"v125_2"], "v125_2_bayes"]

# 9. Benford's Law / Log transformation
# library('BenfordTests')
# for(i in 3:ncol(train)){
#     if(class(all[all[,i] >= 0,i])=='numeric'){
#         sc <- chisq.benftest(all[all[,i] >= 0,i])
#         if(sc$p.value >= 0.05){
#             print(sc)
#             hist(signifd(all[all[,i] >= 0,i]), col = 'red');
#         }else{
#             cat(sc$p.value)
#             hist(signifd(all[all[,i] >= 0,i]), col = 'blue');
#         }
#     }
# }
# 3,8,9,17,20,21,27,28,34,35,39,41,48,52,56,65
# 6,10,12,15,18,19,29,45,57,59,62

# 10. One-hot encoding
# library(caret)
# cate <- names(sapply(all, class)[sapply(all, class) == 'character']); head(all[,c('target',cate[-2])])
# sapply(cate[-2], function(x) length(table(all[,x])))
# 
# for(c in cate[-2]){
#     all[,c] <- as.factor(all[,c])
# }
# dummies <- dummyVars(target ~ ., data = all[,c('target',cate[-2])], sep = "_", levelsOnly = FALSE, fullRank = TRUE)
# all_dum <- as.data.frame(predict(dummies, newdata = all[,c('target',cate[-2])]))
# all_dum <- cbind(all[,!names(all) %in% cate[-2]], all_dum)

load('./BNP-Paribas-Cardif-Claims-Management/meta data/meta_data_20160305.RData')
# 11. Dist
# library(caret) # v22
# train_dum <- all_dum[all_dum$target>=0, ]
# test_dum <- all_dum[all_dum$target<0, ]
# centroids <- classDist(all_dum[, !names(all_dum) %in% c('ID', 'target', 'v22')], as.factor(all_dum[, 'target']),pca = T, keep = 390) # 380
# distances <- predict(centroids, all_dum[, !names(all_dum) %in% c('ID', 'target', 'v22')])
# distances <- as.data.frame(distances)
# distances_all <- distances[,-1]; names(distances_all) <- paste('DistALL', 1:2, sep = "")

# 12. tsne/kmeans
# library(Rtsne)
# tsne <- Rtsne(data.matrix(all_dum[, !names(all_dum) %in% c('ID', 'target', 'v22')]), dims = 3, perplexity=30, check_duplicates = F, pca = F, theta=0.5) #max_iter = 300, 
# embedding <- as.data.frame(tsne$Y)
# tsne_all <- embedding[,1:3]; names(tsne_all) <- c('TSNE_A1','TSNE_A2','TSNE_A3')

# 13. factorize
az_to_int <- function(az) {
    xx <- strsplit(tolower(az), "")[[1]]
    pos <- match(xx, letters[(1:26)]) 
    result <- sum( pos* 26^rev(seq_along(xx)-1))
    return(result)
}

for(f in cate){
    all[,f]<-sapply(all[,f], az_to_int)
}
head(all[,cate])

apply(all[, cate], 2, function(x) mean(is.na(x))); str(all[,cate])
all[, cate][is.na(all[,cate])] <- -999
head(all[,cate])

# 14. Genetic programming to automatically create non-linear features

# 15. Recursive Feature Elimination

# 16. Automation

# 17. AUC / Mean

# 18. Merge
all_comp <- cbind(all, distances_all, tsne_all)
dim(all_comp); head(all_comp)
# save(distances_all, tsne_all, file = './BNP-Paribas-Cardif-Claims-Management/meta data/meta_data_20160305.RData')


#-----------------
# Adding noise ---
#-----------------


#----------
# Split ---
#----------
train <- all_comp[all_comp$target >= 0, ]
test <- all_comp[all_comp$target < 0, ]
save(train, test, file = './data/train_test_20160305.RData')
