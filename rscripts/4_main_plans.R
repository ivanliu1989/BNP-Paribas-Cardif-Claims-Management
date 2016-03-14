setwd('/Users/ivanliu/Downloads/Kaggle_BNP')
library(data.table);library(caret)
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
    # 0.458809
        
# 13.   Benouilli Naive Bayes
    library(e1071)
    cate <- c('v3', 'v24', 'v30', 'v31', 'v47', 'v52','v56', 'v66', 'v71', 'v74', 'v75', 'v79', 'v91107', 'v110', 'v112', 'v113', 'v125',
              'v22_1', 'v22_2', 'v22_3', 'v22_4', 'v56_1', 'v56_2', 'v113_1', 'v113_2', 'v125_1', 'v125_2')
    
    for(f in cate){
        print(f)
        print(table(all[,f]))
        # train the model
        all_dm <- all[,c('target', f)]; all_dm[,f] <- as.factor(all_dm[,f])
        dummies = model.matrix(target~-1+., data = all_dm)
        all_nb_temp <- as.data.frame(cbind(target = all_dm$target, dummies))
        
        train_nb_temp <- all_nb_temp[all_nb_temp$target >= 0, ]
        test_nb_temp <- all_nb_temp[all_nb_temp$target < 0, ]
        # predict
        nb <- naiveBayes(x=train_nb_temp[,-1], y=factor(train_nb_temp$target, labels="x"), laplace = 0)
        all[, paste0(f,'_nb')] <- c(predict(nb, newdata=train_nb_temp, type="raw")[,2], predict(nb, newdata=test_nb_temp, type="raw")[,2])
        
        print(table(all[, paste0(f,'_nb')] ))
    }
    
    # cv_score_13 <- doXGB(train = all[all$target >= 0,!names(all) %in% cate], preproc = FALSE, cv = 5)
    # 0.4591532
    
    train_py <- fread("./data/train_py.csv", stringsAsFactors = F, data.table = F, na.strings = "")
    test_py <- fread("./data/test_py.csv", stringsAsFactors = F, data.table = F, na.strings = "")
    all_nb <- cbind(all[all$target >= 0,!names(all) %in% cate], train_py[, 116:133])
    all_nb <- all_nb[,-c(142:168)]
    cv_score_13_2 <- doXGB(train = all_nb[all_nb$target >= 0,], preproc = FALSE, cv = 5)
    # 0.4585532
    
# 14.   Imputation for non-systematic variables

# 15.   Feature selection mRMR
    head(all)
    all_nb <- cbind(all[all$target >= 0,!names(all) %in% cate], train_py[, 116:133])
    write.csv(all_nb[all_nb$target >= 0,-1], file='train_mRMR.csv', row.names = F)
    # ./mrmr -i ./train_mRMR.csv -n 150 -s 114321 -t 1

    MaxRel_items <- c('ID', 'target', 'v56', 'v79', 'v31_bayes', 'v31', 'v110', 'v110_bayes', 'v62', 'v66', 'v66_bayes',
                    'v113', 'v47', 'v129', 'Cnt_0_row', 'v38', 'TSNE_A3', 'DistALL1', 'v72', 'TSNE_NA_1', 'v30', 'v30_bayes',
                    'v24_bayes', 'v24', 'v71', 'v125_bayes', 'v125', 'v91107', 'v91107_bayes', 'v74', 'TSNE_A2')
    mRMR_items_dp <- c('v11', 'v111', 'v100', 'v1', 'v58', 'v32', 'v73', 'v57', 'v116', 'v27', 'v120', 'v28', 'v43', 'v67', 'v68', 
                    'v60', 'v53', 'v29', 'v45', 'v83', 'v55', 'v33', 'v35', 'v49', 'v99', 'v77', 'v42', 'v26', 'v13', 'v41', 'v6',
                    'v7', 'v96', 'v94')
    
    # cv_score_15_1 <- doXGB(train = all[all$target >= 0,names(all) %in% MaxRel_items], preproc = FALSE, cv = 5)
    # 0.4877296
    # cv_score_15_2 <- doXGB(train = all[all$target >= 0,!names(all) %in% mRMR_items_dp], preproc = FALSE, cv = 5)
    # 
    
    #     *** MaxRel features ***
    #         Order 	 Fea 	 Name 	 Score
    #     1 	 144 	 "v56" 	 0.034
    #     2 	 149 	 "v79" 	 0.029
    #     3 	 123 	 "v31_bayes" 	 0.026
    #     4 	 141 	 "v31" 	 0.025
    #     5 	 152 	 "v110" 	 0.020
    #     6 	 127 	 "v110_bayes" 	 0.020
    #     7 	 55 	 "v62" 	 0.019
    #     8 	 145 	 "v66" 	 0.017
    #     9 	 125 	 "v66_bayes" 	 0.017
    #     10 	 154 	 "v113" 	 0.017
    #     11 	 142 	 "v47" 	 0.013
    #     12 	 111 	 "v129" 	 0.008
    #     13 	 115 	 "Cnt_0_row" 	 0.006
    #     14 	 34 	 "v38" 	 0.005
    #     15 	 120 	 "TSNE_A3" 	 0.005
    #     16 	 116 	 "DistALL1" 	 0.004
    #     17 	 63 	 "v72" 	 0.004
    #     18 	 136 	 "TSNE_NA_1" 	 0.003
    #     19 	 140 	 "v30" 	 0.002
    #     20 	 122 	 "v30_bayes" 	 0.002
    #     21 	 121 	 "v24_bayes" 	 0.001
    #     22 	 139 	 "v24" 	 0.001
    #     23 	 146 	 "v71" 	 0.001
    #     24 	 129 	 "v125_bayes" 	 0.001
    #     25 	 155 	 "v125" 	 0.001
    #     26 	 150 	 "v91" 	 0.001
    #     27 	 151 	 "v107" 	 0.001
    #     28 	 126 	 "v91107_bayes" 	 0.001
    #     29 	 147 	 "v74" 	 0.001
    #     30 	 119 	 "TSNE_A2" 	 0.001
    #     31 	 135 	 "v125_2_bayes" 	 0.000
    #     32 	 118 	 "TSNE_A1" 	 0.000
    #     33 	 128 	 "v112_bayes" 	 0.000
    #     34 	 153 	 "v112" 	 0.000
    #     35 	 138 	 "v3" 	 0.000
    #     36 	 137 	 "TSNE_NA_2" 	 0.000
    #     37 	 20 	 "v21" 	 0.000
    #     38 	 100 	 "v117" 	 0.000
    #     39 	 110 	 "v128" 	 0.000
    #     40 	 70 	 "v82" 	 0.000
    #     41 	 69 	 "v81" 	 0.000
    #     42 	 95 	 "v109" 	 0.000
    #     43 	 94 	 "v108" 	 0.000
    #     44 	 4 	 "v5" 	 0.000
    #     45 	 32 	 "v36" 	 0.000
    #     46 	 62 	 "v70" 	 0.000
    #     47 	 107 	 "v124" 	 0.000
    #     48 	 7 	 "v8" 	 0.000
    #     49 	 56 	 "v63" 	 0.000
    #     50 	 77 	 "v89" 	 0.000
    #     51 	 42 	 "v46" 	 0.000
    #     52 	 48 	 "v54" 	 0.000
    #     53 	 23 	 "v25" 	 0.000
    #     54 	 75 	 "v87" 	 0.000
    #     55 	 85 	 "v98" 	 0.000
    #     56 	 92 	 "v105" 	 0.000
    #     57 	 114 	 "Cnt_NA_row" 	 0.000
    #     58 	 132 	 "v22_3_bayes" 	 0.000
    #     59 	 73 	 "v85" 	 0.000
    #     60 	 102 	 "v119" 	 0.000
    #     61 	 22 	 "v23" 	 0.000
    #     62 	 46 	 "v51" 	 0.000
    #     63 	 106 	 "v123" 	 0.000
    #     64 	 2 	 "v2" 	 0.000
    #     65 	 93 	 "v106" 	 0.000
    #     66 	 88 	 "v101" 	 0.000
    #     67 	 43 	 "v48" 	 0.000
    #     68 	 3 	 "v4" 	 0.000
    #     69 	 16 	 "v17" 	 0.000
    #     70 	 54 	 "v61" 	 0.000
    #     71 	 52 	 "v59" 	 0.000
    #     72 	 40 	 "v44" 	 0.000
    #     73 	 65 	 "v76" 	 0.000
    #     74 	 57 	 "v64" 	 0.000
    #     75 	 19 	 "v20" 	 0.000
    #     76 	 58 	 "v65" 	 0.000
    #     77 	 104 	 "v121" 	 0.000
    #     78 	 8 	 "v9" 	 0.000
    #     79 	 105 	 "v122" 	 0.000
    #     80 	 68 	 "v80" 	 0.000
    #     81 	 109 	 "v127" 	 0.000
    #     82 	 108 	 "v126" 	 0.000
    #     83 	 91 	 "v104" 	 0.000
    #     84 	 90 	 "v103" 	 0.000
    #     85 	 96 	 "v111" 	 0.000
    #     86 	 1 	 "v1" 	 0.000
    #     87 	 28 	 "v32" 	 0.000
    #     88 	 50 	 "v57" 	 0.000
    #     89 	 25 	 "v27" 	 0.000
    #     90 	 26 	 "v28" 	 0.000
    #     91 	 59 	 "v67" 	 0.000
    #     92 	 53 	 "v60" 	 0.000
    #     93 	 27 	 "v29" 	 0.000
    #     94 	 41 	 "v45" 	 0.000
    #     95 	 71 	 "v83" 	 0.000
    #     96 	 49 	 "v55" 	 0.000
    #     97 	 29 	 "v33" 	 0.000
    #     98 	 31 	 "v35" 	 0.000
    #     99 	 44 	 "v49" 	 0.000
    #     100 	 86 	 "v99" 	 0.000
    #     101 	 66 	 "v77" 	 0.000
    #     102 	 38 	 "v42" 	 0.000
    #     103 	 24 	 "v26" 	 0.000
    #     104 	 12 	 "v13" 	 0.000
    #     105 	 37 	 "v41" 	 0.000
    #     106 	 5 	 "v6" 	 0.000
    #     107 	 6 	 "v7" 	 0.000
    #     108 	 83 	 "v96" 	 0.000
    #     109 	 81 	 "v94" 	 0.000
    #     110 	 80 	 "v93" 	 0.000
    #     111 	 72 	 "v84" 	 0.000
    #     112 	 76 	 "v88" 	 0.000
    #     113 	 17 	 "v18" 	 0.000
    #     114 	 74 	 "v86" 	 0.000
    #     115 	 61 	 "v69" 	 0.000
    #     116 	 113 	 "v131" 	 0.000
    #     117 	 67 	 "v78" 	 0.000
    #     118 	 15 	 "v16" 	 0.000
    #     119 	 98 	 "v115" 	 0.000
    #     120 	 14 	 "v15" 	 0.000
    #     121 	 35 	 "v39" 	 0.000
    #     122 	 10 	 "v11" 	 0.000
    #     123 	 87 	 "v100" 	 0.000
    #     124 	 51 	 "v58" 	 0.000
    #     125 	 64 	 "v73" 	 0.000
    #     126 	 99 	 "v116" 	 0.000
    #     127 	 103 	 "v120" 	 0.000
    #     128 	 39 	 "v43" 	 0.000
    #     129 	 60 	 "v68" 	 0.000
    #     130 	 78 	 "v90" 	 0.000
    #     131 	 47 	 "v53" 	 0.000
    #     132 	 84 	 "v97" 	 0.000
    #     133 	 112 	 "v130" 	 0.000
    #     134 	 33 	 "v37" 	 0.000
    #     135 	 18 	 "v19" 	 0.000
    #     136 	 79 	 "v92" 	 0.000
    #     137 	 82 	 "v95" 	 0.000
    #     138 	 101 	 "v118" 	 0.000
    #     139 	 11 	 "v12" 	 0.000
    #     140 	 45 	 "v50" 	 0.000
    #     141 	 9 	 "v10" 	 0.000
    #     142 	 130 	 "v22_1_bayes" 	 0.000
    #     143 	 117 	 "DistALL2" 	 0.000
    #     144 	 89 	 "v102" 	 0.000
    #     145 	 131 	 "v22_2_bayes" 	 0.000
    #     146 	 134 	 "v125_1_bayes" 	 0.000
    #     147 	 13 	 "v14" 	 0.000
    #     148 	 36 	 "v40" 	 0.000
    #     149 	 30 	 "v34" 	 0.000
    #     150 	 143 	 "v52" 	 0.000
    #     
    #     *** mRMR features *** 
    #         Order 	 Fea 	 Name 	 Score
    #     1 	 144 	 "v56" 	 0.034
    #     2 	 132 	 "v22_3_bayes" 	 0.000
    #     3 	 121 	 "v24_bayes" 	 0.001
    #     4 	 11 	 "v12" 	 0.000
    #     5 	 147 	 "v74" 	 0.000
    #     6 	 145 	 "v66" 	 0.001
    #     7 	 97 	 "v114" 	 -0.000
    #     8 	 129 	 "v125_bayes" 	 -0.000
    #     9 	 111 	 "v129" 	 0.000
    #     10 	 143 	 "v52" 	 -0.000
    #     11 	 133 	 "v22_4_bayes" 	 -0.000
    #     12 	 146 	 "v71" 	 -0.001
    #     13 	 149 	 "v79" 	 -0.000
    #     14 	 9 	 "v10" 	 -0.001
    #     15 	 45 	 "v50" 	 -0.001
    #     16 	 150 	 "v91" 	 -0.001
    #     17 	 20 	 "v21" 	 -0.001
    #     18 	 13 	 "v14" 	 -0.002
    #     19 	 138 	 "v3" 	 -0.002
    #     20 	 36 	 "v40" 	 -0.002
    #     21 	 34 	 "v38" 	 -0.002
    #     22 	 61 	 "v69" 	 -0.002
    #     23 	 130 	 "v22_1_bayes" 	 -0.002
    #     24 	 116 	 "DistALL1" 	 -0.002
    #     25 	 30 	 "v34" 	 -0.002
    #     26 	 131 	 "v22_2_bayes" 	 -0.003
    #     27 	 140 	 "v30" 	 -0.003
    #     28 	 154 	 "v113" 	 -0.003
    #     29 	 134 	 "v125_1_bayes" 	 -0.004
    #     30 	 148 	 "v75" 	 -0.006
    #     31 	 120 	 "TSNE_A3" 	 -0.005
    #     32 	 55 	 "v62" 	 -0.006
    #     33 	 117 	 "DistALL2" 	 -0.008
    #     34 	 118 	 "TSNE_A1" 	 -0.009
    #     35 	 119 	 "TSNE_A2" 	 -0.009
    #     36 	 135 	 "v125_2_bayes" 	 -0.009
    #     37 	 115 	 "Cnt_0_row" 	 -0.011
    #     38 	 128 	 "v112_bayes" 	 -0.012
    #     39 	 137 	 "TSNE_NA_2" 	 -0.013
    #     40 	 21 	 "v22" 	 -0.014
    #     41 	 123 	 "v31_bayes" 	 -0.013
    #     42 	 151 	 "v107" 	 -0.018
    #     43 	 63 	 "v72" 	 -0.019
    #     44 	 139 	 "v24" 	 -0.022
    #     45 	 142 	 "v47" 	 -0.022
    #     46 	 122 	 "v30_bayes" 	 -0.023
    #     47 	 124 	 "v52_bayes" 	 -0.022
    #     48 	 125 	 "v66_bayes" 	 -0.025
    #     49 	 136 	 "TSNE_NA_1" 	 -0.030
    #     50 	 141 	 "v31" 	 -0.030
    #     51 	 126 	 "v91107_bayes" 	 -0.030
    #     52 	 155 	 "v125" 	 -0.030
    #     53 	 85 	 "v98" 	 -0.036
    #     54 	 153 	 "v112" 	 -0.038
    #     55 	 127 	 "v110_bayes" 	 -0.041
    #     56 	 89 	 "v102" 	 -0.051
    #     57 	 114 	 "Cnt_NA_row" 	 -0.065
    #     58 	 152 	 "v110" 	 -0.069
    #     59 	 109 	 "v127" 	 -0.079
    #     60 	 75 	 "v87" 	 -0.092
    #     61 	 113 	 "v131" 	 -0.107
    #     62 	 92 	 "v105" 	 -0.119
    #     63 	 2 	 "v2" 	 -0.132
    #     64 	 100 	 "v117" 	 -0.143
    #     65 	 67 	 "v78" 	 -0.156
    #     66 	 62 	 "v70" 	 -0.167
    #     67 	 15 	 "v16" 	 -0.179
    #     68 	 110 	 "v128" 	 -0.188
    #     69 	 8 	 "v9" 	 -0.200
    #     70 	 70 	 "v82" 	 -0.209
    #     71 	 98 	 "v115" 	 -0.220
    #     72 	 69 	 "v81" 	 -0.228
    #     73 	 93 	 "v106" 	 -0.239
    #     74 	 107 	 "v124" 	 -0.247
    #     75 	 105 	 "v122" 	 -0.257
    #     76 	 95 	 "v109" 	 -0.264
    #     77 	 68 	 "v80" 	 -0.274
    #     78 	 94 	 "v108" 	 -0.281
    #     79 	 88 	 "v101" 	 -0.290
    #     80 	 7 	 "v8" 	 -0.296
    #     81 	 84 	 "v97" 	 -0.305
    #     82 	 4 	 "v5" 	 -0.311
    #     83 	 19 	 "v20" 	 -0.319
    #     84 	 32 	 "v36" 	 -0.325
    #     85 	 43 	 "v48" 	 -0.333
    #     86 	 56 	 "v63" 	 -0.339
    #     87 	 112 	 "v130" 	 -0.346
    #     88 	 77 	 "v89" 	 -0.352
    #     89 	 33 	 "v37" 	 -0.359
    #     90 	 42 	 "v46" 	 -0.364
    #     91 	 48 	 "v54" 	 -0.371
    #     92 	 58 	 "v65" 	 -0.376
    #     93 	 23 	 "v25" 	 -0.383
    #     94 	 18 	 "v19" 	 -0.387
    #     95 	 3 	 "v4" 	 -0.394
    #     96 	 104 	 "v121" 	 -0.400
    #     97 	 73 	 "v85" 	 -0.405
    #     98 	 79 	 "v92" 	 -0.411
    #     99 	 22 	 "v23" 	 -0.417
    #     100 	 82 	 "v95" 	 -0.422
    #     101 	 46 	 "v51" 	 -0.428
    #     102 	 101 	 "v118" 	 -0.433
    #     103 	 106 	 "v123" 	 -0.438
    #     104 	 16 	 "v17" 	 -0.443
    #     105 	 102 	 "v119" 	 -0.448
    #     106 	 78 	 "v90" 	 -0.453
    #     107 	 108 	 "v126" 	 -0.458
    #     108 	 54 	 "v61" 	 -0.463
    #     109 	 91 	 "v104" 	 -0.468
    #     110 	 52 	 "v59" 	 -0.473
    #     111 	 90 	 "v103" 	 -0.477
    #     112 	 40 	 "v44" 	 -0.482
    #     113 	 14 	 "v15" 	 -0.486
    #     114 	 65 	 "v76" 	 -0.491
    #     115 	 35 	 "v39" 	 -0.495
    #     116 	 57 	 "v64" 	 -0.499
    
    
    #     117 	 10 	 "v11" 	 -0.503
    #     118 	 96 	 "v111" 	 -0.508
    #     119 	 87 	 "v100" 	 -0.512
    #     120 	 1 	 "v1" 	 -0.516
    #     121 	 51 	 "v58" 	 -0.520
    #     122 	 28 	 "v32" 	 -0.523
    #     123 	 64 	 "v73" 	 -0.527
    #     124 	 50 	 "v57" 	 -0.531
    #     125 	 99 	 "v116" 	 -0.535
    #     126 	 25 	 "v27" 	 -0.538
    #     127 	 103 	 "v120" 	 -0.542
    #     128 	 26 	 "v28" 	 -0.545
    #     129 	 39 	 "v43" 	 -0.549
    #     130 	 59 	 "v67" 	 -0.552
    #     131 	 60 	 "v68" 	 -0.556
    #     132 	 53 	 "v60" 	 -0.559
    #     133 	 47 	 "v53" 	 -0.562
    #     134 	 27 	 "v29" 	 -0.565
    #     135 	 41 	 "v45" 	 -0.569
    #     136 	 71 	 "v83" 	 -0.572
    #     137 	 49 	 "v55" 	 -0.575
    #     138 	 29 	 "v33" 	 -0.578
    #     139 	 31 	 "v35" 	 -0.581
    #     140 	 44 	 "v49" 	 -0.584
    #     141 	 86 	 "v99" 	 -0.586
    #     142 	 66 	 "v77" 	 -0.589
    #     143 	 38 	 "v42" 	 -0.592
    #     144 	 24 	 "v26" 	 -0.595
    #     145 	 12 	 "v13" 	 -0.598
    #     146 	 37 	 "v41" 	 -0.600
    #     147 	 5 	 "v6" 	 -0.603
    #     148 	 6 	 "v7" 	 -0.606
    #     149 	 83 	 "v96" 	 -0.608
    #     150 	 81 	 "v94" 	 -0.611



################
# MODELS: ######
################
# 1.    Vowpal Wabbit: two way or three way interactions

# 2.    Xgboost

# 3.    H2O deep learning and logistic regression

# 4.    LIBFM

# 5.    LIBSVM