# library(devtools)
# install_github("h2oai/h2o-2/R/ensemble/h2oEnsemble-package")
setwd("/Users/ivanliu/Downloads/Kaggle_BNP")
library(h2oEnsemble)
library(cvAUC)

h2o.init(nthreads=3, max_mem_size='10G')

## Load Data into H2O Cluster
train = h2o.uploadFile("./data/train.csv", destination_frame="train.hex")  
train$target = as.factor(train$target)
splits = h2o.splitFrame(train, 0.9, destination_frames=c("trainSplit","testSplit"))

## Base learners
learner = c("h2o.randomForest.wrapper", "h2o.gbm.wrapper")

## Ensemble training
fit = h2o.ensemble(x = 3:133,
                   y = 2, 
                   data = splits[[1]], 
                   family = "binomial", 
                   learner = learner,
                   metalearner ="h2o.glm.wrapper",
                   cvControl = list(V=4)
                   )

## Predict
p = predict(fit, splits[[2]])
labels = as.data.frame(splits[[2]][,"target"])

## Model Evaluation
h2oPredRF  = as.data.frame(p$basepred$h2o.randomForest.wrapper)
h2oPredGBM = as.data.frame(p$basepred$h2o.gbm.wrapper)
h2oPredEns = as.data.frame(p$pred$p1)
# Base learner AUC
AUC(predictions=h2oPredRF, labels=labels)
AUC(predictions=h2oPredGBM, labels=labels)
# h2oEnsemble
AUC(predictions=h2oPredEns, labels=labels)
