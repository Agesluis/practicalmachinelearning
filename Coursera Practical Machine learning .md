---
title: "Practical Machine Learning Coursera"
author: Age Sluis
output: html_document
---

## Introduction  
Please find below my submission for the final assignment of the Practical machine learning course. 

## Data Preprocessing  

```r
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
library(corrplot)
```
## Download the Data

```r
trainUrl <-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
trainFile <- "./data/pml-training.csv"
testFile  <- "./data/pml-testing.csv"
if (!file.exists("./data")) {
  dir.create("./data")
}
if (!file.exists(trainFile)) {
  download.file(trainUrl, destfile=trainFile, method="curl")
}
if (!file.exists(testFile)) {
  download.file(testUrl, destfile=testFile, method="curl")
}
```
## Read the Data

```r
trainRaw <- read.csv("./data/pml-training.csv")
testRaw <- read.csv("./data/pml-testing.csv")
dim(trainRaw)
```

```
[1] 19622   160
```

```r
dim(testRaw)
```

```
[1]  20 160
```
The training data set contains 19622 observations and 160 variables.
The test data set contains 20 observations and 160 variables. 
The "classe" variable in the training set is the outcome to predict.

## Cleaning of the data

```r
sum(complete.cases(trainRaw))
```

```
[1] 406
```
First, remove columns that contain NA missing values.

```r
trainRaw <- trainRaw[, colSums(is.na(trainRaw)) == 0]
testRaw <- testRaw[, colSums(is.na(testRaw)) == 0]
```
Next, get rid of some columns that do not contribute much to the accelerometer measurements.

```r
classe <- trainRaw$classe
trainRemove <- grepl("^X|timestamp|window", names(trainRaw))
trainRaw <- trainRaw[, !trainRemove]
trainCleaned <- trainRaw[, sapply(trainRaw, is.numeric)]
trainCleaned$classe <- classe
testRemove <- grepl("^X|timestamp|window", names(testRaw))
testRaw <- testRaw[, !testRemove]
testCleaned <- testRaw[, sapply(testRaw, is.numeric)]
```
The training data set contains 19622 observations and 53 variables, while the test data set contains 20 observations and 53 variables. The "classe" variable is still in the cleaned training set.

## Training and validation set

Split the cleaned training set into a pure training data set (70%) and a validation data set (30%). 
We will use the validation data set to conduct cross validation in future steps.  

```r
set.seed(20000) # For reproducibile purpose
inTrain <- createDataPartition(trainCleaned$classe, p=0.70, list=F)
trainData <- trainCleaned[inTrain, ]
testData <- trainCleaned[-inTrain, ]
```

## Data Modeling

We fit a predictive model for activity recognition using **Random Forest** algorithm because it automatically selects important variables and is robust to correlated covariates & outliers in general. We will use **5-fold cross validation** when applying the algorithm.  

```r
controlRf <- trainControl(method="cv", 5)
modelRf <- train(classe ~ ., data=trainData, method="rf", trControl=controlRf, ntree=250)
modelRf
```

```
Random Forest

13737 samples
52 predictor
5 classes: 'A', 'B', 'C', 'D', 'E'

No pre-processing
Resampling: Cross-Validated (5 fold)

Summary of sample sizes: 10990, 10991, 10989, 10988, 10990

Resampling results across tuning parameters:

 mtry  Accuracy   Kappa    
   2    0.9913370  0.9890410
  27    0.9903182  0.9877527
  52    0.9846397  0.9805687

Accuracy was used to select the optimal model using  the largest value.
The final value used for the model was mtry = 27.
```
Then, we estimate the performance of the model on the validation data set.  

```r
predictRf <- predict(modelRf, testData)
confusionMatrix(testData$classe, predictRf)
```

```
Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 1671    3    0    0    0
         B    6 1132    1    0    0
         C    0    8 1016    2    0
         D    0    0   27  936    1
         E    0    0    5    0 1077
    
   Overall Statistics
                                           
                     Accuracy : 0.991           
                 95% CI : (0.9882, 0.9932)
    No Information Rate : 0.285           
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.9886          
 Mcnemar's Test P-Value : NA              
             
 
  Statistics by Class:
 
                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            0.9964   0.9904   0.9685   0.9979   0.9991
Specificity            0.9993   0.9985   0.9979   0.9943   0.9990
Pos Pred Value         0.9982   0.9939   0.9903   0.9710   0.9954
Neg Pred Value         0.9986   0.9977   0.9932   0.9996   0.9998
Prevalence             0.2850   0.1942   0.1782   0.1594   0.1832
Detection Rate         0.2839   0.1924   0.1726   0.1590   0.1830
Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
Balanced Accuracy      0.9979   0.9945   0.9832   0.9961   0.9990
```

```r
accuracy <- postResample(predictRf, testData$classe)
accuracy
```

```
Accuracy     Kappa 
0.9909941 0.9886071 
```

```r
oose <- 1 - as.numeric(confusionMatrix(testData$classe, predictRf)$overall[1])
oose
```

```
[1] 0.009005947
```
The accuracy of the model is 99.1% and the out-of-sample error is 0.90%.

## Predictions for Test Data Set
After applying the model to the test set we found that in 100% of the cases the model made the right prediction.  

```r
result <- predict(modelRf, testCleaned[, -length(names(testCleaned))])
result
```

```
[1] B A B A A E D B A A B C B A E E A B B B
Levels: A B C D E
```
