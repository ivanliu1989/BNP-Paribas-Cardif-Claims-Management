---
title: "Using the Boruta Package to Determine Feature Relevance"
author: "Jim Thompson"
date: "`r Sys.time()`"
output: html_document
---

This report determines what features may be relevant to modeling the **target**
attribute for the BNP Paribas Cardif competition.  This analysis is based on
the [**Boruta**](https://cran.r-project.org/web/packages/Boruta/index.html) package.
In addition we demonstrate the [**caret**](https://cran.r-project.org/web/packages/caret/index.html) package functions 
to create data partitions and impute missing values.

First, we load required libraries.
```{r warning=FALSE,message=FALSE}
# required libraries
library(caret)
library(Boruta)
library(dplyr)
```

Read in the training data set, determine data types and perform simple pre-processing 
for numeric data.
```{r}
DATA.DIR <- "../input"

# retrive sample data for analysis
train <- read.csv(paste0(DATA.DIR,"/train.csv"),stringsAsFactors = FALSE)

###
# select random sample for analysis using caret createDataPartition() function
###
set.seed(123)
idx <- createDataPartition(train$target,p=0.01,list=FALSE)
sample.df <- train[idx,]

###
# segregate numeric vs character data types
###
# get names of the explanatory variables
explanatory.attributes <- setdiff(names(sample.df),c("ID","target"))

# determine data type for each explanatory variable
data.classes <- sapply(explanatory.attributes,function(x){class(sample.df[,x])})

# segregate explanatory variables by data type, eg. character, numeric, integer
unique.classes <- unique(data.classes)
attr.by.data.types <- lapply(unique.classes,function(x){names(data.classes[data.classes==x])})
names(attr.by.data.types) <- unique.classes
comment(attr.by.data.types) <- "list that categorize training data types"


# for numeric attributes use caret preProcess() and predict() functions to impute missing values
pp <- preProcess(sample.df[c(attr.by.data.types$numeric,attr.by.data.types$integer)],
                 method=c("medianImpute"))
pp.sample.df <- predict(pp,sample.df[c(attr.by.data.types$numeric,attr.by.data.types$integer)])

# combine numeric data with character data
df <- cbind(pp.sample.df,sample.df[attr.by.data.types$character])

```


Analyze sample data with Boruta package to determine relevance of the explanatory
variables.
```{r}
set.seed(13)
bor.results <- Boruta(df,factor(sample.df$target),
                   maxRuns=101,
                   doTrace=0)
```


```{r echo=FALSE}
cat("\nsummary of Boruta run:\n")
print(bor.results)

cat("\n\nAttributes determined to be relevant:\n")
getSelectedAttributes(bor.results)
```

The following plot shows the relevance of each feature.  Green color indicate
the feature has been deemed relevance to the **target** attribute.  Red color
indicates the attribute is not relevant.  Yellow attributes are tentative, may or 
may not be relevant.  Increasing maxRuns parameter can help further refine
the list of tentative attributes as relevant or not.
```{r echo=FALSE,fig.width=9,fig.height=7}
plot(bor.results)
```

Detailed results for each explanatory results
```{r, echo=FALSE}
cat("\n\nAttribute importance details:\n")
options(width=125)
arrange(cbind(attr=rownames(attStats(bor.results)), attStats(bor.results)),desc(medianImp))
```

