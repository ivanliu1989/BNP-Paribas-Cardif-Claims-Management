### Examine NAs using mice and VIM packages ###
library(data.table)
setwd('/Users/ivanliu/Downloads/Kaggle_BNP')
rm(list = ls()); gc(reset = TRUE)

train <- fread("./data/train.csv", stringsAsFactors = F, data.table = F, na.strings = "")
test <- fread("./data/test.csv", stringsAsFactors = F, data.table = F, na.strings = "")

# read file and convert char to int
library(readr)
for (f in names(train)) {
    if (class(train[[f]])=="character") {
        levels <- unique(train[[f]])
        train[[f]] <- as.integer(factor(train[[f]], levels=levels))
    }
}

# make a table of missing values
# library(mice)
# missers <- md.pattern(train[, -c(1:2)])
# head(missers)
# write_csv(as.data.frame(missers),"NAsTable.csv")

# make plots of missing values
library(VIM)

png(filename="NAsPatternEq.png",
    type="cairo",
    units="in",
    width=12,
    height=6.5,
    pointsize=10,
    res=300)

miceplot1 <- aggr(train[, -c(1:2)], col=c("dodgerblue","dimgray"),
                  numbers=TRUE, combined=TRUE, varheight=FALSE, border="gray50",
                  sortVars=TRUE, sortCombs=FALSE, ylabs=c("Missing Data Pattern"),
                  labels=names(train[-c(1:2)]), cex.axis=.7)
dev.off()

png(filename="NAsPatternAdj.png",
    type="cairo",
    units="in",
    width=12,
    height=6.5,
    pointsize=10,
    res=300)

miceplot2 <- aggr(train[, -c(1:2)], col=c("dodgerblue","dimgray"),
                  numbers=TRUE, combined=TRUE, varheight=TRUE, border=NA,
                  sortVars=TRUE, sortCombs=FALSE, ylabs=c("Missing Data Pattern w/ Height Adjustment"),
                  labels=names(train[-c(1:2)]), cex.axis=.7)
dev.off()



# NA pattern - unsupervised learning
all <- rbind(train[,3:ncol(train)], test[,-1])
head(all)
all[!is.na(all)] <- 1
all[is.na(all)] <- -1
head(all)

# pca
# pca_fit <- prcomp(data.matrix(all))
# plot(pca_fit$x[,1:2],col = km_fit$cluster)

# tsne
library(Rtsne)
tsne <- Rtsne(data.matrix(all), dims = 2, perplexity=30, check_duplicates = F, pca = F, theta=0.5) #max_iter = 300, 
embedding <- as.data.frame(tsne$Y)
tsne_na <- embedding[,1:2]; names(tsne_na) <- c('TSNE_NA_1','TSNE_NA_2')
# kmeans
km_fit <- kmeans(data.matrix(all), centers = 8, iter.max = 100000, nstart = 50)
km_cluster_all <- km_fit$cluster
km_fit <- kmeans(data.matrix(tsne_na), centers = 8, iter.max = 100000, nstart = 50)
km_cluster_tsne <- km_fit$cluster

plot(tsne_na, col = km_cluster_all)
points(km_fit$centers, type = 'o')

save(tsne_na, km_cluster_all, km_cluster_tsne, file = './BNP-Paribas-Cardif-Claims-Management/meta data/na_meta_data_20160307.RData')
