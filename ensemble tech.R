####################problem 1 ##########################
# Load the Data
# Diabetes.csv
Diabetes = read.csv(file.choose())

##Exploring and preparing the data 
str(Diabetes)

library(CatEncoders)

lb <- LabelEncoder.fit(Diabetes$Class.variable)
Class.variable <- transform(lb, Diabetes$Class.variable)
Diabetes1 <- cbind(Diabetes, Class.variable)
Diabetes2 <- Diabetes1[, -9]

library(caTools)
set.seed(0)
split <- sample.split(Diabetes2$Class.variable, SplitRatio = 0.8)
Diabetes_train <- subset(Diabetes2, split == TRUE)
Diabetes_test <- subset(Diabetes2, split == FALSE)

#################bagging#################
# install.packages("randomForest")
library(randomForest)

bagging <- randomForest(Diabetes_train$Class.variable ~ ., data = Diabetes_train, mtry = 8)
# bagging will take all the columns ---> mtry = all the attributes

test_pred <- predict(bagging, Diabetes_test)

rmse_bagging <- sqrt(mean(Diabetes_test$Class.variable - test_pred)^2)
rmse_bagging

# Prediction for trained data result
train_pred <- predict(bagging, Diabetes_train)

# RMSE on Train Data
train_rmse <- sqrt(mean(Diabetes_train$Class.variable - train_pred)^2)
train_rmse

summary(Diabetes_train)

# install.packages("adabag")
library(adabag)

Diabetes_train$Class.variable <- as.factor(Diabetes_train$Class.variable)

adaboost <- boosting(Class.variable ~ ., data = Diabetes_train, boos = TRUE)

# Test data
adaboost_test <- predict(adaboost, Diabetes_test)

table(adaboost_test$class, Diabetes_test$Class.variable)
mean(adaboost_test$class == Diabetes_test$Class.variable)


# Train data
adaboost_train <- predict(adaboost, Diabetes_train)

table(adaboost_train$class, Diabetes_train$Class.variable)
mean(adaboost_train$class == Diabetes_train$Class.variable)

summary(Diabetes_train)
attach(Diabetes_train)

# install.packages("xgboost")
library(xgboost)

train_y <- Diabetes_train$Class.variable == "1"

str(Diabetes_train)

# create dummy variables on attributes
train_x <- model.matrix(Diabetes_train$Class.variable ~ . -1, data = Diabetes_train)

train_x <- train_x[, -9]
# 'n-1' dummy variables are required, hence deleting the additional variables

test_y <- Diabetes_test$Class.variable == "1"

# create dummy variables on attributes
test_x <- model.matrix(Diabetes_test$Class.variable ~ .-1, data = Diabetes_test)
test_x <- test_x[, -9]

# DMatrix on train
Xmatrix_train <- xgb.DMatrix(data = train_x, label = train_y)
# DMatrix on test 
Xmatrix_test <- xgb.DMatrix(data = test_x, label = test_y)


# Max number of boosting iterations - nround
xg_boosting <- xgboost(data = Xmatrix_train, nround = 50,
                       objective = "multi:softmax", eta = 0.3, 
                       num_class = 2, max_depth = 100)

# Prediction for test data
xgbpred_test <- predict(xg_boosting, Xmatrix_test)
table(test_y, xgbpred_test)
mean(test_y == xgbpred_test)

# Prediction for train data
xgbpred_train <- predict(xg_boosting, Xmatrix_train)
table(train_y, xgbpred_train)
mean(train_y == xgbpred_train)

# Voting for Classification

# load dataset with factors as strings
Diabetes <- read.csv(file.choose(), stringsAsFactors = TRUE)
str(Diabetes)
set.seed(12345)
Train_Test <- sample(c("Train", "Test"), nrow(Diabetes), replace = TRUE, prob = c(0.7, 0.3))
Diabetes_Train <- Diabetes[Train_Test == "Train",]
Diabetes_TestX <- within(Diabetes[Train_Test == "Test", ], rm(Class.variable))
Diabetes_TestY <- Diabetes[Train_Test == "Test", "Class.variable"]

library(randomForest)
# Random Forest Analysis
Diabetes_RF <- randomForest(Class.variable ~ ., data = Diabetes_Train, keep.inbag = TRUE, ntree = 500)

# Overall class prediction (hard voting)
Diabetes_RF_Test_Margin <- predict(Diabetes_RF, newdata = Diabetes_TestX, type = "class")

# Prediction
Diabetes_RF_Test_Predict <- predict(Diabetes_RF, newdata = Diabetes_TestX, type = "class", predict.all = TRUE)

sum(Diabetes_RF_Test_Margin == Diabetes_RF_Test_Predict$aggregate)
head(Diabetes_RF_Test_Margin == Diabetes_RF_Test_Predict$aggregate)

# Majority Voting
dim(Diabetes_RF_Test_Predict$individual)

# View(cc_RF_Test_Predict$individual) # Prediction at each tree

Row_Count_Max <- function(x) names(which.max(table(x)))

Voting_Predict <- apply(Diabetes_RF_Test_Predict$individual, 1, Row_Count_Max)

head(Voting_Predict)
tail(Voting_Predict)

all(Voting_Predict == Diabetes_RF_Test_Predict$aggregate)
all(Voting_Predict == Diabetes_RF_Test_Margin)

mean(Voting_Predict == Diabetes_TestY)

########################problem 2###########################
# Load the Data
# Tumor_Ensemble.csv

tumor <- read.csv(file.choose())

###removing id column##
tumor <- tumor[-1]

# Exploratory Data Analysis

# table of diagnosis
table(tumor$diagnosis)

str(tumor$diagnosis)
# recode diagnosis as a factor
tumor$diagnosis <- factor(tumor$diagnosis, levels = c("B", "M"), labels = c("Benign", "Malignant"))

# table or proportions with more informative labels
round(prop.table(table(tumor$diagnosis)) * 100, digits = 2)


library(caTools)
set.seed(0)
split <- sample.split(tumor$diagnosis, SplitRatio = 0.8)
tumor_train <- subset(tumor, split == TRUE)
tumor_test <- subset(tumor, split == FALSE)


# install.packages("randomForest")
library(randomForest)

bagging <- randomForest(tumor_train$diagnosis ~ ., data = tumor_train, mtry = 8)
# bagging will take all the columns ---> mtry = all the attributes

test_pred <- predict(bagging, tumor_test)

rmse_bagging <- sqrt(mean(tumor_test$diagnosis - test_pred)^2)
rmse_bagging

# Prediction for trained data result
train_pred <- predict(bagging, tumor_train)

# RMSE on Train Data
train_rmse <- sqrt(mean(tumor_train$diagnosis - train_pred)^2)
train_rmse

summary(tumor_train)

# install.packages("adabag")
library(adabag)

tumor_train$diagnosis <- as.factor(tumor_train$diagnosis)

adaboost <- boosting(diagnosis ~ ., data = tumor_train, boos = TRUE)

# Test data
adaboost_test <- predict(adaboost, tumor_test)

table(adaboost_test$class, tumor_test$diagnosis)
mean(adaboost_test$class == tumor_test$diagnosis)


# Train data
adaboost_train <- predict(adaboost, tumor_train)

table(adaboost_train$class, tumor_train$diagnosis)
mean(adaboost_train$class == tumor_train$diagnosis)

summary(tumor_train)
attach(tumor_train)

# install.packages("xgboost")
library(xgboost)

train_y <- tumor_train$diagnosis == "1"

str(tumor_train)

# create dummy variables on attributes
train_x <- model.matrix(tumor_train$diagnosis ~ . -1, data = tumor_train)

train_x <- train_x[, -1]
# 'n-1' dummy variables are required, hence deleting the additional variables

test_y <- as.matrix(tumor_test$Class.variable == "1")

# create dummy variables on attributes
test_x <- model.matrix(tumor_test$diagnosis ~ .-1, data = tumor_test)
test_x <- test_x[, -1]

# DMatrix on train
Xmatrix_train <- xgb.DMatrix(data = train_x, label = train_y)
# DMatrix on test 
Xmatrix_test <- xgb.DMatrix(data = test_x, label = test_y)

xgboost::xgb.DMatrix(data=x, label=mat_y)

# Max number of boosting iterations - nround
xg_boosting <- xgboost(data = Xmatrix_train, nround = 50,
                       objective = "multi:softmax", eta = 0.3, 
                       num_class = 2, max_depth = 100)

# Prediction for test data
xgbpred_test <- predict(xg_boosting, Xmatrix_test)
table(test_y, xgbpred_test)
mean(test_y == xgbpred_test)

# Prediction for train data
xgbpred_train <- predict(xg_boosting, Xmatrix_train)
table(train_y, xgbpred_train)
mean(train_y == xgbpred_train)

# Voting for Classification

# load dataset with factors as strings
tumors <- read.csv(file.choose(), stringsAsFactors = TRUE)
str(tumors)
set.seed(12345)
Train_Test <- sample(c("Train", "Test"), nrow(tumors), replace = TRUE, prob = c(0.7, 0.3))
tumors_Train <- tumors[Train_Test == "Train",]
tumors_TestX <- within(tumors[Train_Test == "Test", ], rm(diagnosis))
tumors_TestY <- tumors[Train_Test == "Test", "diagnosis"]

library(randomForest)
# Random Forest Analysis
tumors_RF <- randomForest(diagnosis ~ ., data = tumors_Train, keep.inbag = TRUE, ntree = 500)

# Overall class prediction (hard voting)
tumors_RF_Test_Margin <- predict(tumors_RF, newdata = tumors_TestX, type = "class")

# Prediction
tumors_RF_Test_Predict <- predict(tumors_RF, newdata = tumors_TestX, type = "class", predict.all = TRUE)

sum(tumors_RF_Test_Margin == tumors_RF_Test_Predict$aggregate)
head(tumors_RF_Test_Margin == tumors_RF_Test_Predict$aggregate)

# Majority Voting
dim(tumors_RF_Test_Predict$individual)

# View(cc_RF_Test_Predict$individual) # Prediction at each tree

Row_Count_Max <- function(x) names(which.max(table(x)))

Voting_Predict <- apply(tumors_RF_Test_Predict$individual, 1, Row_Count_Max)

head(Voting_Predict)
tail(Voting_Predict)

all(Voting_Predict == tumors_RF_Test_Predict$aggregate)
all(Voting_Predict == tumors_RF_Test_Margin)

mean(Voting_Predict == tumors_TestY)
#####################problem 3###############################
install.packages("readxl")
library(readxl)

cocoa1 <- read_excel(file.choose())
##Exploring and preparing the data
str(cocoa1)
sum(is.na(cocoa1))
cocoa1<- na.omit(cocoa1)
str(cocoa1)

cocoa1 <- cocoa1[,-3]
cocoa1 <- cocoa1[,-7]
cocoa1 <- cocoa1[,-3]
cocoa1 <-cocoa1[,-6]
str(cocoa1)

names(cocoa1)
cocoa1$Name <- tolower(gsub(pattern = '[[:space:]+]', '_', cocoa1$Name))
cocoa1$Company <- tolower(gsub(pattern = '[[:space:]+]', '_', cocoa1$Company))
cocoa1$Company_Location <- tolower(gsub(pattern = '[[:space:]+]', '_', cocoa1$Company_Location))

cocoa1$Name <- tolower(gsub(pattern = ',', '_', cocoa1$Name))
cocoa1$Company <- tolower(gsub(pattern = ',', '_', cocoa1$Company))
cocoa1$Company_Location <- tolower(gsub(pattern = ',', '_', cocoa1$Company_Location))

cocoa1 <- cocoa1[,-2]

library(CatEncoders)

lb <- LabelEncoder.fit(cocoa1$Company)
Company <- transform(lb, cocoa1$Company)
cocoa1 <- cbind(cocoa1, Company)
cocoa1 <- cocoa1[, -1]


lb <- LabelEncoder.fit(cocoa1$Company_Location)
Company_Location <- transform(lb, cocoa1$Company_Location)
cocoa1 <- cbind(cocoa1, Company_Location)
cocoa1 <- cocoa1[, -2]

cocoa1$Rating = ifelse(cocoa1$Rating<3, "No", "Yes")
cocoa1$Rating <- factor(cocoa1$Rating, levels = c("Yes", "No"), labels = c("good", "bad"))
str(cocoa1)

library(caret)
local <- createDataPartition(cocoa1$Rating, p = 0.75, list = F)
training <- cocoa1[local, ]
testing <- cocoa1[-local, ]
str(training)


library(C50)
model1 <- C5.0(training$Rating~ ., data = training[, -5])
plot(model1)
pred <- predict.C5.0(model1, testing[, -5])
a <- table(testing$Rating, pred)

sum(diag(a))/sum(a)

##############################################################################
#Voting
#install.packages("readxl")
library(readxl)

cocoa1 <- read_excel(file.choose())
##Exploring and preparing the data ----
str(cocoa1)
sum(is.na(cocoa1))
cocoa1<- na.omit(cocoa1)
str(cocoa1)

cocoa1 <- cocoa1[,-3]
cocoa1 <- cocoa1[,-7]
cocoa1 <- cocoa1[,-3]
cocoa1 <-cocoa1[,-6]
str(cocoa1)

names(cocoa1)
cocoa1$Name <- tolower(gsub(pattern = '[[:space:]+]', '_', cocoa1$Name))
cocoa1$Company <- tolower(gsub(pattern = '[[:space:]+]', '_', cocoa1$Company))
cocoa1$Company_Location <- tolower(gsub(pattern = '[[:space:]+]', '_', cocoa1$Company_Location))

cocoa1$Name <- tolower(gsub(pattern = ',', '_', cocoa1$Name))
cocoa1$Company <- tolower(gsub(pattern = ',', '_', cocoa1$Company))
cocoa1$Company_Location <- tolower(gsub(pattern = ',', '_', cocoa1$Company_Location))

cocoa1 <- cocoa1[,-2]

library(CatEncoders)

lb <- LabelEncoder.fit(cocoa1$Company)
Company <- transform(lb, cocoa1$Company)
cocoa1 <- cbind(cocoa1, Company)
cocoa1 <- cocoa1[, -1]


lb <- LabelEncoder.fit(cocoa1$Company_Location)
Company_Location <- transform(lb, cocoa1$Company_Location)
cocoa1 <- cbind(cocoa1, Company_Location)
cocoa1 <- cocoa1[, -2]

cocoa1$Rating = ifelse(cocoa1$Rating<3, "No", "Yes")
cocoa1$Rating <- factor(cocoa1$Rating, levels = c("Yes", "No"), labels = c("good", "bad"))
str(cocoa1)


set.seed(101)
Train_Test <- sample(c("Train", "Test"), nrow(cocoa1), replace = TRUE, prob = c(0.7, 0.3))
b1_Train <- cocoa1[Train_Test == "Train",]
b1_TestX <- within(cocoa1[Train_Test == "Test", ], rm(Rating))
b1_TestY <- cocoa1[Train_Test == "Test", "Rating"]


library(randomForest)
# Random Forest Analysis
b1_RF <- randomForest(Rating ~ ., data = b1_Train, keep.inbag = TRUE, ntree = 500)
?randomForest

# Overall class prediction (hard voting)
b1_RF_Test_Margin <- predict(b1_RF, newdata = b1_TestX, type = "class")

# Prediction
b1_RF_Test_Predict <- predict(b1_RF, newdata = b1_TestX, type = "class", predict.all = TRUE)

sum(b1_RF_Test_Margin == b1_RF_Test_Predict$aggregate)
head(b1_RF_Test_Margin == b1_RF_Test_Predict$aggregate)

# Majority Voting
dim(b1_RF_Test_Predict$individual)

# View(cc_RF_Test_Predict$individual) # Prediction at each tree

Row_Count_Max <- function(x) names(which.max(table(x)))

Voting_Predict <- apply(b1_RF_Test_Predict$individual, 1, Row_Count_Max)

head(Voting_Predict)
tail(Voting_Predict)

all(Voting_Predict == b1_RF_Test_Predict$aggregate)
all(Voting_Predict == b1_RF_Test_Margin)

mean(Voting_Predict == b1_TestY)

#########################################################################################
#Bagging

install.packages("readxl")
library(readxl)

cocoa1 <- read_excel(file.choose())
##Exploring and preparing the data 
str(cocoa1)
sum(is.na(cocoa1))
cocoa1<- na.omit(cocoa1)
str(cocoa1)

cocoa1 <- cocoa1[,-3]
cocoa1 <- cocoa1[,-7]
cocoa1 <- cocoa1[,-3]
cocoa1 <-cocoa1[,-6]
str(cocoa1)

names(cocoa1)
cocoa1$Name <- tolower(gsub(pattern = '[[:space:]+]', '_', cocoa1$Name))
cocoa1$Company <- tolower(gsub(pattern = '[[:space:]+]', '_', cocoa1$Company))
cocoa1$Company_Location <- tolower(gsub(pattern = '[[:space:]+]', '_', cocoa1$Company_Location))

cocoa1$Name <- tolower(gsub(pattern = ',', '_', cocoa1$Name))
cocoa1$Company <- tolower(gsub(pattern = ',', '_', cocoa1$Company))
cocoa1$Company_Location <- tolower(gsub(pattern = ',', '_', cocoa1$Company_Location))

cocoa1 <- cocoa1[,-2]

library(CatEncoders)

lb <- LabelEncoder.fit(cocoa1$Company)
Company <- transform(lb, cocoa1$Company)
cocoa1 <- cbind(cocoa1, Company)
cocoa1 <- cocoa1[, -1]


lb <- LabelEncoder.fit(cocoa1$Company_Location)
Company_Location <- transform(lb, cocoa1$Company_Location)
cocoa1 <- cbind(cocoa1, Company_Location)
cocoa1 <- cocoa1[, -2]


library(caTools)
set.seed(0)
split <- sample.split(cocoa1$Rating, SplitRatio = 0.8)
d1_train <- subset(cocoa1, split == TRUE)
d1_test <- subset(cocoa1, split == FALSE)


# install.packages("randomForest")
library(randomForest)

bagging <- randomForest(d1_train$Rating ~ ., data = d1_train)

print(bagging)
plot(bagging)
# bagging will take all the columns ---> mtry = all the attributes

test_pred <- predict(bagging, d1_test)

rmse_bagging <- sqrt(mean(d1_test$Rating - test_pred)^2)
rmse_bagging

# Prediction for trained data result
train_pred <- predict(bagging, d1_train)

# RMSE on Train Data
train_rmse <- sqrt(mean(d1_train$Rating - train_pred)^2)
train_rmse

###############################################################################
#Boosting
#Adaboost
library(readxl)

cocoa1 <- read_excel(file.choose())
##Exploring and preparing the data 
str(cocoa1)
sum(is.na(cocoa1))
cocoa1<- na.omit(cocoa1)
str(cocoa1)

cocoa1 <- cocoa1[,-3]
cocoa1 <- cocoa1[,-7]
cocoa1 <- cocoa1[,-3]
cocoa1 <-cocoa1[,-6]
str(cocoa1)

names(cocoa1)
cocoa1$Name <- tolower(gsub(pattern = '[[:space:]+]', '_', cocoa1$Name))
cocoa1$Company <- tolower(gsub(pattern = '[[:space:]+]', '_', cocoa1$Company))
cocoa1$Company_Location <- tolower(gsub(pattern = '[[:space:]+]', '_', cocoa1$Company_Location))

cocoa1$Name <- tolower(gsub(pattern = ',', '_', cocoa1$Name))
cocoa1$Company <- tolower(gsub(pattern = ',', '_', cocoa1$Company))
cocoa1$Company_Location <- tolower(gsub(pattern = ',', '_', cocoa1$Company_Location))

cocoa1 <- cocoa1[,-2]

library(CatEncoders)

lb <- LabelEncoder.fit(cocoa1$Company)
Company <- transform(lb, cocoa1$Company)
cocoa1 <- cbind(cocoa1, Company)
cocoa1 <- cocoa1[, -1]


lb <- LabelEncoder.fit(cocoa1$Company_Location)
Company_Location <- transform(lb, cocoa1$Company_Location)
cocoa1 <- cbind(cocoa1, Company_Location)
cocoa1 <- cocoa1[, -2]

cocoa1$Rating = ifelse(cocoa1$Rating<3, "No", "Yes")
cocoa1$Rating <- factor(cocoa1$Rating, levels = c("Yes", "No"), labels = c("good", "bad"))
str(cocoa1)



library(caTools)
set.seed(0)
split <- sample.split(cocoa1$Rating, SplitRatio = 0.8)
d1_train <- subset(cocoa1, split == TRUE)
d1_test <- subset(cocoa1, split == FALSE)

summary(d1_train)

install.packages("adabag")
library(adabag)


adaboost <- boosting(Rating ~ ., data = d1_train, boos = TRUE)

# Test data
adaboost_test <- predict(adaboost, d1_test)

table(adaboost_test$class, d1_test$Rating)
mean(adaboost_test$class == d1_test$Rating)


# Train data
adaboost_train <- predict(adaboost, d1_train)

table(adaboost_train$class, d1_train$Rating)
mean(adaboost_train$class == d1_train$Rating)

#########################################################################
# Gradient boosting

library(readxl)

cocoa1 <- read_excel(file.choose())
##Exploring and preparing the data ----
str(cocoa1)
sum(is.na(cocoa1))
cocoa1<- na.omit(cocoa1)
str(cocoa1)

cocoa1 <- cocoa1[,-3]
cocoa1 <- cocoa1[,-7]
cocoa1 <- cocoa1[,-3]
cocoa1 <-cocoa1[,-6]
str(cocoa1)

names(cocoa1)
cocoa1$Name <- tolower(gsub(pattern = '[[:space:]+]', '_', cocoa1$Name))
cocoa1$Company <- tolower(gsub(pattern = '[[:space:]+]', '_', cocoa1$Company))
cocoa1$Company_Location <- tolower(gsub(pattern = '[[:space:]+]', '_', cocoa1$Company_Location))

cocoa1$Name <- tolower(gsub(pattern = ',', '_', cocoa1$Name))
cocoa1$Company <- tolower(gsub(pattern = ',', '_', cocoa1$Company))
cocoa1$Company_Location <- tolower(gsub(pattern = ',', '_', cocoa1$Company_Location))

cocoa1 <- cocoa1[,-2]

library(CatEncoders)

lb <- LabelEncoder.fit(cocoa1$Company)
Company <- transform(lb, cocoa1$Company)
cocoa1 <- cbind(cocoa1, Company)
cocoa1 <- cocoa1[, -1]


lb <- LabelEncoder.fit(cocoa1$Company_Location)
Company_Location <- transform(lb, cocoa1$Company_Location)
cocoa1 <- cbind(cocoa1, Company_Location)
cocoa1 <- cocoa1[, -2]


library(caTools)
set.seed(0)
split <- sample.split(cocoa1$Rating, SplitRatio = 0.8)
d1_train <- subset(cocoa1, split == TRUE)
d1_test <- subset(cocoa1, split == FALSE)


# install.packages("gbm")
library(gbm)

boosting <- gbm(d1_train$Rating ~ ., data = d1_train, distribution = 'gaussian',
                n.trees = 5000, interaction.depth = 4, shrinkage = 0.2, verbose = F)
# distribution = Gaussian for regression and Bernoulli for classification

plot(boosting)
# Prediction for test data result
boost_test <- predict(boosting, d1_test, n.trees = 5000)

rmse_boosting <- sqrt(mean(d1_test$Rating - boost_test)^2)
rmse_boosting

# Prediction for train data result
boost_train <- predict(boosting, d1_train, n.trees = 5000)

# RMSE on Train Data
rmse_train <- sqrt(mean(d1_train$Rating - boost_train)^2)
rmse_train

#################################################################################
#Xgboost

library(readxl)

cocoa1 <- read_excel(file.choose())
##Exploring and preparing the data ----
str(cocoa1)
sum(is.na(cocoa1))
cocoa1<- na.omit(cocoa1)
str(cocoa1)

cocoa1 <- cocoa1[,-3]
cocoa1 <- cocoa1[,-7]
cocoa1 <- cocoa1[,-3]
cocoa1 <-cocoa1[,-6]
str(cocoa1)

names(cocoa1)
cocoa1$Name <- tolower(gsub(pattern = '[[:space:]+]', '_', cocoa1$Name))
cocoa1$Company <- tolower(gsub(pattern = '[[:space:]+]', '_', cocoa1$Company))
cocoa1$Company_Location <- tolower(gsub(pattern = '[[:space:]+]', '_', cocoa1$Company_Location))

cocoa1$Name <- tolower(gsub(pattern = ',', '_', cocoa1$Name))
cocoa1$Company <- tolower(gsub(pattern = ',', '_', cocoa1$Company))
cocoa1$Company_Location <- tolower(gsub(pattern = ',', '_', cocoa1$Company_Location))

cocoa1 <- cocoa1[,-2]

library(CatEncoders)

lb <- LabelEncoder.fit(cocoa1$Company)
Company <- transform(lb, cocoa1$Company)
cocoa1 <- cbind(cocoa1, Company)
cocoa1 <- cocoa1[, -1]


lb <- LabelEncoder.fit(cocoa1$Company_Location)
Company_Location <- transform(lb, cocoa1$Company_Location)
cocoa1 <- cbind(cocoa1, Company_Location)
cocoa1 <- cocoa1[, -2]

cocoa1$Rating = ifelse(cocoa1$Rating<3, "No", "Yes")
cocoa1$Rating <- factor(cocoa1$Rating, levels = c("Yes", "No"), labels = c("good", "bad"))
str(cocoa1)

library(caTools)
set.seed(0)
split <- sample.split(cocoa1$Rating, SplitRatio = 0.8)
d1_train <- subset(cocoa1, split == TRUE)
d1_test <- subset(cocoa1, split == FALSE)

summary(d1_train)
attach(d1_train)

# install.packages("xgboost")
library(xgboost)

train_y <- d1_train$Rating == "good"

str(d1_train)

# create dummy variables on attributes
train_x <- model.matrix(d1_train$Rating ~ . -1, data = d1_train)

# 'n-1' dummy variables are required, hence deleting the additional variables

test_y <- d1_test$Rating == "good"

# create dummy variables on attributes
test_x <- model.matrix(d1_test$Rating ~ .-1, data = d1_test)

# DMatrix on train
Xmatrix_train <- xgb.DMatrix(data = train_x, label = train_y)
# DMatrix on test 
Xmatrix_test <- xgb.DMatrix(data = test_x, label = test_y)


# Max number of boosting iterations - nround
xg_boosting <- xgboost(data = Xmatrix_train, nround = 50,
                       objective = "multi:softmax", eta = 0.3, 
                       num_class = 2, max_depth = 100)

# Prediction for test data
xgbpred_test <- predict(xg_boosting, Xmatrix_test)
table(test_y, xgbpred_test)
mean(test_y == xgbpred_test)

# Prediction for train data
xgbpred_train <- predict(xg_boosting, Xmatrix_train)
table(train_y, xgbpred_train)
mean(train_y == xgbpred_train)
##################### problem 4############################
# Load the Data
# Ensemble_password_strength.csv

library(readxl)
password <-read_excel(file.choose())


# Exploratory Data Analysis

# table of diagnosis
table(password$characters_strength)

str(password$characters_strength)


library(caTools)
set.seed(0)
split <- sample.split(password$characters_strength, SplitRatio = 0.8)
password_train <- subset(password, split == TRUE)
password_test <- subset(password, split == FALSE)


# install.packages("randomForest")
library(randomForest)

bagging <- randomForest(password_train$characters_strength ~ ., data = password_train, mtry = 6)
# bagging will take all the columns ---> mtry = all the attributes

test_pred <- predict(bagging, password_test)

rmse_bagging <- sqrt(mean(password_test$characters_strength - test_pred)^2)
rmse_bagging

# Prediction for trained data result
train_pred <- predict(bagging, password_train)

# RMSE on Train Data
train_rmse <- sqrt(mean(password_train$characters_strength - train_pred)^2)
train_rmse

summary(password_train)

# install.packages("adabag")
library(adabag)

password_train$characters_strength <- as.factor(password_train$characters_strength)

adaboost <- boosting(characters_strength ~ ., data = password_train, boos = TRUE)

# Test data
adaboost_test <- predict(adaboost, password_test)

table(adaboost_test$class, password_test$characters_strength)
mean(adaboost_test$class == password_test$characters_strength)


# Train data
adaboost_train <- predict(adaboost, password_train)

table(adaboost_train$class, password_train$characters_strength)
mean(adaboost_train$class == password_train$characters_strength)

summary(password_train)
attach(password_train)

# install.packages("xgboost")
library(xgboost)

train_y <- password_train$characters_strength == "1"

str(password_train)

# create dummy variables on attributes
train_x <- model.matrix(password_train$characters_strength~ . -1, data = password_train)

train_x <- train_x[, -2]
# 'n-1' dummy variables are required, hence deleting the additional variables

test_y <- as.matrix(password_test$characters_strength == "1")

# create dummy variables on attributes
test_x <- model.matrix(password_test$characters_strength ~ .-2, data = tumor_test)
test_x <- test_x[, -2]

# DMatrix on train
Xmatrix_train <- xgb.DMatrix(data = train_x, label = train_y)
# DMatrix on test 
Xmatrix_test <- xgb.DMatrix(data = test_x, label = test_y)

xgboost::xgb.DMatrix(data=x, label=mat_y)

# Max number of boosting iterations - nround
xg_boosting <- xgboost(data = Xmatrix_train, nround = 50,
                       objective = "multi:softmax", eta = 0.3, 
                       num_class = 2, max_depth = 100)

# Prediction for test data
xgbpred_test <- predict(xg_boosting, Xmatrix_test)
table(test_y, xgbpred_test)
mean(test_y == xgbpred_test)

# Prediction for train data
xgbpred_train <- predict(xg_boosting, Xmatrix_train)
table(train_y, xgbpred_train)
mean(train_y == xgbpred_train)

# Voting for Classification

# load dataset with factors as strings
library(readxl)
passwords <- read_excel(file.choose())
str(passwords)
set.seed(12345)
passwords_Test <- sample(c("Train", "Test"), nrow(passwords), replace = TRUE, prob = c(0.7, 0.3))
passwords_Train <- passwords[Train_Test == "Train",]
passwords_TestX <- within(passwords[Train_Test == "Test", ], rm(characters_strength))
passwords_TestY <- passwords[Train_Test == "Test", "characters_strength"]

library(randomForest)
# Random Forest Analysis
passwords_RF <- randomForest(characters_strength ~ ., data = passwords_Train, keep.inbag = TRUE, ntree = 500)

# Overall class prediction (hard voting)
passwords_RF_Test_Margin <- predict(passwords_RF, newdata = passwords_TestX, type = "class")

# Prediction
passwords_RF_Test_Predict <- predict(passwords_RF, newdata = passwords_TestX, type = "class", predict.all = TRUE)

sum(passwords_RF_Test_Margin == passwords_RF_Test_Predict$aggregate)
head(passwords_RF_Test_Margin == passwords_RF_Test_Predict$aggregate)

# Majority Voting
dim(passwords_RF_Test_Predict$individual)


Row_Count_Max <- function(x) names(which.max(table(x)))

Voting_Predict <- apply(passwords_RF_Test_Predict$individual, 1, Row_Count_Max)

head(Voting_Predict)
tail(Voting_Predict)

all(Voting_Predict == passwords_RF_Test_Predict$aggregate)
all(Voting_Predict == passwords_RF_Test_Margin)

mean(Voting_Predict == passwords_TestY)
