library(caret)

# 1) -------------------------------------------------------------------------------

library(AppliedPredictiveModeling)

data(AlzheimerDisease)
adData <- data.frame(diagnosis, predictors)
inTrain <- createDataPartition(diagnosis, p = 0.5, list = FALSE)
train <- adData[inTrain,]
test <- adData[-inTrain,]

# 2) -------------------------------------------------------------------------------

library(AppliedPredictiveModeling)
library(Hmisc)

data(concrete)
set.seed(1000)
inTrain = createDataPartition(mixtures$CompressiveStrength,  p = 0.5, list = FALSE)
training = mixtures[ inTrain,]
testing = mixtures[-inTrain,]

predictoras <- colnames(concrete)[1:8]

# ver correlación de cada predictora con la dependiente: no hay ninguna evidente
featurePlot(x = training[, predictoras], y = training$CompressiveStrength, plot = "pairs")

# crear columna de índice
training$index <- seq.int(nrow(training))

# plotear índice vs dependiente: se ve patrón escalonado
ggplot(data = training, aes(x = index, y = CompressiveStrength)) + geom_point()
# lo mismo
qplot(index, CompressiveStrength, data = training)

# colorear plot por predictora, ver si hay relación entre escalones y la predictora => no hay (probar otros breaks?)
qplot(Cement, CompressiveStrength, data = training, colour = cut2(training$Age, g = 10))
qplot(BlastFurnaceSlag, CompressiveStrength, data = training, colour = cut2(training$Age, g = 10))
qplot(FlyAsh, CompressiveStrength, data = training, colour = cut2(training$Age, g = 10))
qplot(Water, CompressiveStrength, data = training, colour = cut2(training$Age, g = 10))
qplot(Superplasticizer, CompressiveStrength, data = training, colour = cut2(training$Age, g = 10))
qplot(CoarseAggregate, CompressiveStrength, data = training, colour = cut2(training$Age, g = 10))
qplot(FineAggregate, CompressiveStrength, data = training, colour = cut2(training$Age, g = 10))
qplot(Age, CompressiveStrength, data = training, colour = cut2(training$Age, g = 10))

# mirar una predictora para ver sus cortes
# hist(training$FlyAsh)
# training$FlyAsh.discretizada <- cut2(training$FlyAsh, g = 10)
# qplot(x = FlyAsh, y = CompressiveStrength, colour = FlyAsh.discretizada, data = training)

# 3) -------------------------------------------------------------------------------

library(AppliedPredictiveModeling
library(caret)

data(concrete)
set.seed(1000)
inTrain = createDataPartition(mixtures$CompressiveStrength, p = 3/4)[[1]]
training = mixtures[ inTrain,]
testing = mixtures[-inTrain,]

# la de nros negativos no puede ser, porque no hay
which(training$Superplasticizer < 0)

# confirmar skewness
hist(training$Superplasticizer)

# la transformación log(_+1) no sirve porque casi todos los valores están en un bucket al principio, y seguirían estando ahí.
hist(log(training$Superplasticizer+1))


# 4) -------------------------------------------------------------------------------

library(caret)
library(AppliedPredictiveModeling)

set.seed(3433)
data(AlzheimerDisease)
adData = data.frame(diagnosis,predictors)
inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]]
training = adData[ inTrain,]
testing = adData[-inTrain,]

# Find all the predictor variables in the training set that begin with IL. 
predictoras.IL <- grep("^[Ii][Ll].*", names(training))

# Perform principal components on these variables with the preProcess() function from the caret package.
# threshold es el corte de varianza acumulada explicada por pca
training.pca <- preProcess(training[, predictoras.IL], method=c("center", "scale", "pca"), thresh = 0.8)
training.pca

## Created from 251 samples and 12 variables
## Pre-processing:
##   - centered (12)
##   - ignored (0)
##   - principal component signal extraction (12)
##   - scaled (12)
## PCA needed 7 components to capture 80 percent of the variance

# Calculate the number of principal components needed to capture 80% of the variance. How many are there?

# 5) -------------------------------------------------------------------------------

library(caret)
library(AppliedPredictiveModeling)

set.seed(3433)
data(AlzheimerDisease)
adData = data.frame(diagnosis, predictors)
inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]]
training = adData[inTrain,]
testing = adData[-inTrain,]

# Create a training data set consisting of only the predictors with variable names beginning with IL and the diagnosis. 

set.seed(3433)
IL <- grep("^il", colnames(training), ignore.case = TRUE, value = TRUE)
adData.IL <- data.frame(diagnosis, predictors[, predictoras.IL])
inTrain.IL <- createDataPartition(adData.IL$diagnosis, p = 3/4)[[1]]
training.IL <- adData.IL[inTrain.IL,]
testing.IL <- adData.IL[-inTrain.IL,]

# Build two predictive models, one using the predictors as they are and one using PCA with principal components explaining 80% of the variance in the predictors. Use method="glm" in the train function.

modelo.como_vienen <- train(diagnosis ~ ., method = "glm", data = training.IL)
predicciones.como_vienen <- predict(modelo.como_vienen, newdata = testing.IL)
confusionMatrix.como_vienen <- confusionMatrix(predicciones.como_vienen, testing.IL$diagnosis)
accuracy.como_vienen <- confusionMatrix.como_vienen$overall[1]
accuracy.como_vienen
# 0.6463415

modelo.pca <- train(diagnosis ~ ., method = "glm", preProcess = "pca", data = training.IL, trControl = trainControl(preProcOptions = list(thresh = 0.8)))
predicciones.pca <- predict(modelo.pca, newdata = testing.IL)
confusionMatrix.pca <- confusionMatrix(predicciones.pca, testing.IL$diagnosis)
accuracy.pca <- confusionMatrix.pca$overall[1]
accuracy.pca
# 0.7195122

# --------------------------------------------------------------------------------
