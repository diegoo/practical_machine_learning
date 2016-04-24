library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
library(corrplot)

library(parallel)
library(doParallel)
clusters <- makeCluster(detectCores() - 1)
registerDoParallel(clusters)

## load data

training.file <- "data/pml-training.csv"
testing.file <- "data/pml-testing.csv"
training <- read.csv(training.file, na.strings = c("NA", "", "#DIV/0!"), header = TRUE, sep = ',')
testing <- read.csv(testing.file, na.strings = c("NA", "", "#DIV/0!"), header = TRUE, sep = ',')

## clean data

training.nona <- training[, colSums(is.na(training)) == 0] 
testing.nona <- testing[, colSums(is.na(testing)) == 0] 

remove.indices <- grepl("user_name|^X|timestamp|window", names(training.nona))
training.clean <- training.nona[, !remove.indices]
remove.indices <- grepl("user_name|^X|timestamp|window", names(testing.nona))
testing.clean <- testing.nona[, !remove.indices]

## partition train data

set.seed(12345)
train.indices <- createDataPartition(training.clean$classe, p = 0.70, list = FALSE)
train.data <- training.clean[train.indices,]
test.data <- training.clean[-train.indices,]

## build model

train.control <- trainControl(method = "cv", 5, classProbs = TRUE, savePredictions = TRUE, allowParallel = TRUE)
system.time(model.rf <- train(classe ~ ., data = train.data, method = "rf", trControl = train.control, ntree = 250))
model.rf
stopCluster(clusters)

## Random Forest 
## 13737 samples
##    52 predictors
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 10989, 10990, 10990, 10989, 10990 
## Resampling results across tuning parameters:
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
##    2    0.9914101  0.9891330  0.001679589  0.002124719
##   27    0.9900270  0.9873842  0.001166909  0.001475135
##   52    0.9862415  0.9825951  0.001442765  0.001824660
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 2. 

varImp(model.rf)

##   only 20 most important variables shown (out of 52)
##                   Overall
## roll_belt          100.00
## yaw_belt            83.55
## magnet_dumbbell_y   64.93
## magnet_dumbbell_z   64.32
## pitch_forearm       61.05
## pitch_belt          59.91
## magnet_dumbbell_x   50.96
## roll_forearm        49.25
## accel_belt_z        46.40
## accel_dumbbell_y    44.24
## magnet_belt_y       41.60
## roll_dumbbell       40.50
## magnet_belt_z       40.09
## accel_dumbbell_z    37.79
## roll_arm            35.95
## accel_forearm_x     32.55
## gyros_belt_z        29.57
## accel_dumbbell_x    29.34
## magnet_arm_y        28.26
## yaw_dumbbell        27.86

predictions.rf <- predict(modelRf, test.data)
summary(predictions.rf)
confusionMatrix(test.data$classe, predictions.rf)

## Confusion Matrix and Statistics
##           Reference
## Prediction    A    B    C    D    E
##          A 1672    2    0    0    0
##          B   12 1121    6    0    0
##          C    0   19 1003    4    0
##          D    0    0   27  937    0
##          E    0    0    1    3 1078
## Overall Statistics
##                Accuracy : 0.9874          
##                  95% CI : (0.9842, 0.9901)
##     No Information Rate : 0.2862          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                   Kappa : 0.9841          
##  Mcnemar's Test P-Value : NA              
## Statistics by Class:
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9929   0.9816   0.9672   0.9926   1.0000
## Specificity            0.9995   0.9962   0.9953   0.9945   0.9992
## Pos Pred Value         0.9988   0.9842   0.9776   0.9720   0.9963
## Neg Pred Value         0.9972   0.9956   0.9930   0.9986   1.0000
## Prevalence             0.2862   0.1941   0.1762   0.1604   0.1832
## Detection Rate         0.2841   0.1905   0.1704   0.1592   0.1832
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9962   0.9889   0.9812   0.9936   0.9996

## accuracy <- postResample(predictRf, testData$classe)
## accuracy
##  Accuracy     Kappa 
## 0.9874257 0.9840906

# lo mismo:
out.of.sample.accuracy <- sum(predictions.rf == test.data$classe) / length(predictions.rf)
out.of.sample.accuracy

## [1] 0.9874257

out.of.sample.error <- 1 - out.of.sample.accuracy
out.of.sample.error

## lo mismo:
## out.of.sample.error <- 1 - as.numeric(confusionMatrix(testData$classe, predictRf)$overall[1])
## out.of.sample.error

## [1] 0.01257434

paste0("Out of sample error estimation: ", round(out.of.sample.error * 100, digits = 2), "%")

## apply model & predict

results <- predict(model.rf, testing[, -length(names(testing))])
results

## [1] B A B A A E D B A A B C B A E E A B B B

## write file with predictions

pml_write_files = function(x) {
    n = length(x)
    for (i in 1:n) {
        filename = paste0("problem_id_", i, ".txt")
        write.table(x[i], file = filename, quote = FALSE, row.names = FALSE, col.names = FALSE)
    }
}

pml_write_files(results)

### --------------------------------------------------------------------------------

## preprocess with PCA

train.data.no.classe <- train.data[,!(names(train.data) %in% c("classe"))]
train.data.pca <- preProcess(train.data.no.classe, method = "pca", pcaComp = 5)

### --------------------------------------------------------------------------------

training.clean.nzv <- nearZeroVar(training.clean, saveMetrics = TRUE)
if (any(training.clean.nzv$nzv)) nzv else "all predictors have some variance"

### --------------------------------------------------------------------------------

model.rpart <- rpart(classe ~ ., data = train.data, method = "class")
fancyRpartPlot(model.rpart)
predictions.rpart <- predict(model.rpart, test.data, type = "class")
confusionMatrix(predictions.rpart, test.data$classe)

## Confusion Matrix and Statistics
##           Reference
## Prediction    A    B    C    D    E
##          A 1498  196   69  106   25
##          B   42  669   85   86   92
##          C   43  136  739  129  131
##          D   33   85   98  553   44
##          E   58   53   35   90  790
## Overall Statistics
##                Accuracy : 0.722           
##                  95% CI : (0.7104, 0.7334)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                   Kappa : 0.6467          
##  Mcnemar's Test P-Value : < 2.2e-16       
## Statistics by Class:
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8949   0.5874   0.7203  0.57365   0.7301
## Specificity            0.9060   0.9357   0.9097  0.94717   0.9509
## Pos Pred Value         0.7909   0.6869   0.6273  0.68020   0.7700
## Neg Pred Value         0.9559   0.9043   0.9390  0.91897   0.9399
## Prevalence             0.2845   0.1935   0.1743  0.16381   0.1839
## Detection Rate         0.2545   0.1137   0.1256  0.09397   0.1342
## Detection Prevalence   0.3218   0.1655   0.2002  0.13815   0.1743
## Balanced Accuracy      0.9004   0.7615   0.8150  0.76041   0.8405



## plot features

total <- which(grepl("^total", colnames(trainingData), ignore.case = F))
totalAccel <- trainingData[, total]
featurePlot(x = totalAccel, y = trainingData$classe, pch = 19, main = "Feature plot", plot = "pairs")

## visualize model

plot(modelRf, log = "y", lwd = 2, main = "Random forest accuracy", xlab = "Predictors", ylab = "Accuracy")
treeModel <- rpart(classe ~ ., data=trainData, method="class")
prp(treeModel)

## --------------------------------------------------------------------------------

library(parallel)
library(doParallel)
clusters <- makeCluster(detectCores() - 1)
registerDoParallel(clusters)

## --------------------------------------------------------------------------------

## preprocessing no hace falta cuando se usan modelos no-parametricos como rf (week 2, "basic preprocessing" lecture)

## --------------------------------------------------------------------------------

varImpPlot(model.rf$finalModel)

## --------------------------------------------------------------------------------

