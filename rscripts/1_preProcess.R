library(data.table)
setwd("/Users/ivanliu/Downloads/Kaggle_BNP")
rm(list = ls()); gc(reset = TRUE)

train <- fread("./data/train.csv", stringsAsFactors = F, data.table = F, na.strings = "")
test <- fread("./data/test.csv", stringsAsFactors = F, data.table = F)
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



#------------------------
# Create New features ---
#------------------------
# 1. Counts of NA
Cnt_NA_row

# 2. Categorical variables: v91 - v107

# 3. Categorical variables: v71 - v75

# 4. Categorical variables: v79 - v71

# 5. Categorical variables: v10 - v31

# 6. Continuous variables

# 7. Mean target for category?

# 8. Dist

# 9. tsne/kmeans


#----------------------------
# Feature transformations ---
#----------------------------


#-----------------
# Adding noise ---
#-----------------


#----------
# Split ---
#----------
