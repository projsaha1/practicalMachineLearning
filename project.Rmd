---
title: 'Practical Machine Learning: Assignment'
author: "Projjwal Saha"
date: "February 17, 2019"
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```
## Project and Goal
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, the goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

The training data for this project are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The goal of the project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. Other variables in the train dataset are used to predict the classe variable.

## Data Analysis
Firstly both the train and test dataset were downloaded to the machine where the ML algorithms would run. For this case, the files are download in Windows file system.

```{r, exec=FALSE, results='hide'}
library(caret)
rawdata_train = read.csv("D:/Work/DS/MLProject/pml-training.csv")
summary(rawdata_train)
```
The data consists of 19622 observations with 160 variables. Some columns like 'kurtosis_yaw_belt', 'skewness_roll_belt'  have majority of blank or NA values. If the data is observed carefully, these columns have blank/NA values for rows which have "new_Window" value as "yes". On looking further we see that about 460 rows have "new_window" as yes. These rows have other column values pretty similar to rows with "new_window" as "no". Since these 460 rows constitute less than 3% of the total data, it would be good if we delete such rows and then we can remove all the columns with all blank or NA values.

## Data preprocessing

Here are the steps that are followed - 

1. Remove rows with "new_window" as "yes"
2. Apply preProcess method of caret library to filter columns with near zero variance.
3. Remove columns with all values as blank/NA.
4. Drop columns which are not required for predicting classe like 'user_name', 'raw_timestamp_part_1' and so on.

```{r}
index = which(rawdata_train["new_window"] == "yes")
data_train_omitted = rawdata_train[-index,]
preprocessParams = preProcess(data_train_omitted, method = "nzv")
data_train_transformed = predict(preprocessParams, data_train_omitted)
func_train <- function(x){
  isTRUE(all(x==""))
}
result = apply(data_train_transformed, 2, func_train)
data_train_transformed_1 = data_train_transformed[, !names(data_train_transformed) %in% names(result[result == TRUE])]

dropcols <- c('X', 'user_name', 'raw_timestamp_part_1', 'raw_timestamp_part_2', 'cvtd_timestamp', 'new_window', 'num_window')
data_train <- data_train_transformed_1[, !names(data_train_transformed_1) %in% dropcols]
dim(data_train)
```

As we see, the final dataset "data_train" has 19216 observations with 53 columns. So we have trimmed down unwanted columns from the dataset to give better results from ML training.

We apply the same method to the test data set and obtain the final data_test.

```{r}
rawdata_test = read.csv("D:/Work/DS/MLProject/pml-testing.csv")
data_test_transformed = predict(preprocessParams, rawdata_test)
func_test <- function(x){
  isTRUE(all(is.na(x)))
}
result_test = apply(data_test_transformed, 2, func_test)
data_test_transformed_1 = data_test_transformed[, !names(data_test_transformed) %in% names(result_test[result_test == TRUE])]
dropcols_test <- c('X', 'user_name', 'raw_timestamp_part_1', 'raw_timestamp_part_2', 'cvtd_timestamp', 'new_window', 'num_window', 'problem_id')
data_test <- data_test_transformed_1[, !names(data_test_transformed_1) %in% dropcols_test]
```


## Training/Model Selection

First approach is to use decision trees to predict "classe" from other variables.

```{r}
model_tree <- train(classe ~ ., data = data_train, method = "rpart")
print(model_tree)
```

This gives an accuracy of 50% which is not impressive. Adding cross validation to decision trees didnt help either

```{r}
model_tree_cv <- train(classe ~ ., data = data_train, method = "rpart", trControl = trainControl(method= "cv"))
print(model_tree_cv)
```

Next random forest method is tried. On using default paramters, the code executed endlessly with no output. This proved that control paramters need to be used for rf method. Two control paramters were tried.
1. mtry - Number of variables randomly sampled as candidates at each split.
2. ntree - Number of trees to grow.

So first attempt is to use ntree = 10 and mtry to be related to sqrt of number of columns.

```{r}
control <- trainControl(method='repeatedcv', 
                        number=10, 
                        repeats=3)
metric <- "Accuracy"
mtry <- sqrt(ncol(data_train))
tunegrid <- expand.grid(.mtry=mtry)
model_rf <- train(classe~., 
                    data=data_train, 
                    method='rf', 
                    metric='Accuracy', 
                    tuneGrid=tunegrid, 
                    trControl=control,
                    ntree = 10)
print(model_rf)
```

This accuracy of this model is 99.1%.
On trying another model with ntree=100, gave slightly better accuracy of 99.6%
```{r}
model_rf_2 <- train(classe~., 
                    data=data_train, 
                    method='rf', 
                    metric='Accuracy', 
                    tuneGrid=tunegrid, 
                    trControl=control,
                    ntree = 100)


print(model_rf_2)
```

## Prediction
model_rf_2 gives acceptable accuracy of 99.6%. So using this model to predict the outcomes from test data, predict function is used.

```{r}
pred_rf_2 <- predict(model_rf_2, newdata = data_test)
print(pred_rf_2)
```

