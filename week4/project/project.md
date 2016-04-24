---
title: Coursera | Practical Machine Learning | Project
  Report
author: "Diego Dell'Era"
output:
  html_document:
    fig_height: 10
    fig_width: 10
---

Coursera | Practical Machine Learning | Project
===============================================

## Goal

Use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants to predict the manner in which they did an exercise. 

## Load dependencies


```r
library(caret)
library(rpart)
library(randomForest)

library(parallel)
library(doParallel)
clusters <- makeCluster(detectCores() - 1)
registerDoParallel(clusters)
```

## Load data


```r
training.file <- "data/pml-training.csv"
testing.file <- "data/pml-testing.csv"
training <- read.csv(training.file, na.strings = c("NA", "", "#DIV/0!"), header = TRUE, sep = ',')
testing <- read.csv(testing.file, na.strings = c("NA", "", "#DIV/0!"), header = TRUE, sep = ',')
```

## Clean data


```r
training.nona <- training[, colSums(is.na(training)) == 0] 
testing.nona <- testing[, colSums(is.na(testing)) == 0] 

remove.indices <- grepl("user_name|^X|timestamp|window", names(training.nona))
training.clean <- training.nona[, !remove.indices]
remove.indices <- grepl("user_name|^X|timestamp|window", names(testing.nona))
testing.clean <- testing.nona[, !remove.indices]
```

## Partition training data


```r
set.seed(12345)
train.indices <- createDataPartition(training.clean$classe, p = 0.70, list = FALSE)
train.data <- training.clean[train.indices,]
test.data <- training.clean[-train.indices,]
```

## Build models

### Decision Tree


```r
model.rpart <- rpart(classe ~ ., data = train.data, method = "class")
```

Errors are scattered over all classes:


```r
predictions.rpart <- predict(model.rpart, test.data, type = "class")
confusionmatrix.rpart <- confusionMatrix(predictions.rpart, test.data$classe)
confusionmatrix.rpart$table
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 1498  196   69  106   25
##          B   42  669   85   86   92
##          C   43  136  739  129  131
##          D   33   85   98  553   44
##          E   58   53   35   90  790
```


```r
out.of.sample.accuracy.rpart <- sum(predictions.rpart == test.data$classe) / length(predictions.rpart)
paste0("Decision Tree: Out of sample accuracy estimation: ", round(out.of.sample.accuracy.rpart * 100, digits = 2), "%")
```

```
## [1] "Decision Tree: Out of sample accuracy estimation: 72.2%"
```

```r
out.of.sample.error.rpart <- 1 - out.of.sample.accuracy.rpart
paste0("Decision Tree: Out of sample error estimation: ", round(out.of.sample.error.rpart * 100, digits = 2), "%")
```

```
## [1] "Decision Tree: Out of sample error estimation: 27.8%"
```

72%. Not too bad, but we can try other models.

### Random Forest


```r
train.control <- trainControl(method = "cv", number = 5, classProbs = TRUE, savePredictions = TRUE, allowParallel = TRUE)
model.rf <- train(classe ~ ., data = train.data, method = "rf", trControl = train.control, ntree = 250)
stopCluster(clusters)
predictions.rf <- predict(model.rf, test.data)
confusionmatrix.rf <- confusionMatrix(test.data$classe, predictions.rf)
```

This model is much better:


```r
out.of.sample.accuracy.rf <- sum(predictions.rf == test.data$classe) / length(predictions.rf)
paste0("Random Forest: Out of sample accuracy estimation: ", round(out.of.sample.accuracy.rf * 100, digits = 2), "%")
```

```
## [1] "Random Forest: Out of sample accuracy estimation: 98.93%"
```

```r
out.of.sample.error.rf <- 1 - out.of.sample.accuracy.rf
paste0("Random Forest: Out of sample error estimation: ", round(out.of.sample.error.rf * 100, digits = 2), "%")
```

```
## [1] "Random Forest: Out of sample error estimation: 1.07%"
```

## Apply winning model


```r
final.predictions <- predict(model.rf, testing[, -length(names(testing))])
final.predictions
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

## Write predictions to files


```r
pml_write_files = function(x) {
    n = length(x)
    for (i in 1:n) {
        filename = paste0("predictions/problem_id_", i, ".txt")
        write.table(x[i], file = filename, quote = FALSE, row.names = FALSE, col.names = FALSE)
    }
}

pml_write_files(final.predictions)
```
