## 1.

library(AppliedPredictiveModeling)
data(segmentationOriginal)
library(caret)

## 1. Subset the data to a training set and testing set based on the Case variable in the data set.
## 2. Set the seed to 125 and fit a CART model with the rpart method using all predictor variables and default caret settings.
## 3. In the final model what would be the final model prediction for cases with the following variable values:
## a. TotalIntench2 = 23,000; FiberWidthCh1 = 10; PerimStatusCh1=2
## b. TotalIntench2 = 50,000; FiberWidthCh1 = 10;VarIntenCh4 = 100
## c. TotalIntench2 = 57,000; FiberWidthCh1 = 8;VarIntenCh4 = 100
## d. FiberWidthCh1 = 8;VarIntenCh4 = 100; PerimStatusCh1=2

training <- segmentationOriginal[segmentationOriginal$Case == "Train",]
testing <- segmentationOriginal[segmentationOriginal$Case == "Test",]

set.seed(125)
model <- train(Class ~ ., method = "rpart", data = training)

## model$finalModel
##
## n= 1009 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 1) root 1009 373 PS (0.63032706 0.36967294)  
##   2) TotalIntenCh2< 45323.5 454  34 PS (0.92511013 0.07488987) *
##   3) TotalIntenCh2>=45323.5 555 216 WS (0.38918919 0.61081081)  
##     6) FiberWidthCh1< 9.673245 154  47 PS (0.69480519 0.30519481) *
##     7) FiberWidthCh1>=9.673245 401 109 WS (0.27182045 0.72817955) *

plot(model$finalModel)
text(model$finalModel)

## a. PS
## b. WS
## c. PS
## d. no se puede predecir


## 2.

## The bias is larger and the variance is smaller. Under leave one out cross validation K is equal to the sample size.


## 3.

library(pgmm)
data(olive)
olive = olive[,-1]

model.olive <- train(Area ~ ., method = "rpart", data = olive)

newdata = as.data.frame(t(colMeans(olive)))
predict(model.olive, newdata = newdata)

## 2.783

levels(as.factor(olive$Area))
## [1] "1" "2" "3" "4" "5" "6" "7" "8" "9"
## hay 9 regiones; 2.7 no es una región!

## 2.783. It is strange because Area should be a qualitative variable - but tree is reporting the average value of Area as a numeric variable in the leaf predicted for newdata


## 4.

library(ElemStatLearn)
data(SAheart)
set.seed(8484)
train = sample(1:dim(SAheart)[1],size=dim(SAheart)[1]/2,replace=F)
trainSA = SAheart[train,]
testSA = SAheart[-train,]

missClass = function(values,prediction){sum(((prediction > 0.5)*1) != values)/length(values)}

## Then set the seed to 13234 and fit a logistic regression model
## (method="glm", be sure to specify family="binomial") with Coronary
## Heart Disease (chd) as the outcome and age at onset, current
## alcohol consumption, obesity levels, cumulative tabacco, type-A
## behavior, and low density lipoprotein cholesterol as
## predictors

set.seed(13234)

model.heart <- train(chd ~ age + alcohol + obesity + tobacco + typea + ldl, method = "glm", family = "binomial", data = trainSA)

missClass(trainSA$chd, predict(model.heart, newdata = trainSA))
## [1] 0.2727273
missClass(testSA$chd, predict(model.heart, newdata = testSA))
## [1] 0.3116883


## 5.

library(ElemStatLearn)
data(vowel.train)
data(vowel.test)

vowel.train$y.f <- as.factor(vowel.train$y)
vowel.test$y.f <- as.factor(vowel.test$y)

model.vowel <- train(y.f ~ x.1 + x.2 + x.3 + x.4 + x.5 + x.6 + x.7 + x.8 + x.9 + x.10, data = vowel.train, method = "rf", importance = FALSE)
    
set.seed(33833)
varImp(model.vowel)

## rf variable importance
##      Overall
## x.1  100.000
## x.2   99.955
## x.5   42.021
## x.6   29.574
## x.8   20.067
## x.4   13.006
## x.9    9.602
## x.3    7.426
## x.7    2.224
## x.10   0.000

## la opción más parecida entre las respuestas posibles es: x.2, x.1, x.5, x.6, x.8, x.4, x.9, x.3, x.7,x.10
