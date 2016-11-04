rm(list = ls())                                                       #clear all

require(ROCR)                                                         #essential package for visualisation

#set destination location
setwd ("/Users/dvmvnds/Desktop/R Projects/")

#essential efficiency function
findBestThold <- function( predictions, labels, figFileName ){
  # pred <- prediction(predictions, testing$good)
  pred <- prediction(predictions, labels)
  # next apply code from PredictionMetrics.R
  perf <- performance(pred,measure="acc",x.measure="cutoff")
  
  # Now let's get the cutoff for the best accuracy
  bestAccInd <- which.max(perf@"y.values"[[1]])
  bestAccuracy <- round(perf@"y.values"[[1]][bestAccInd], 4)
  bestThold <- round(perf@"x.values"[[1]][bestAccInd], 4)
  bestMsg <- print(paste("best accuracy=", bestAccuracy,
                         " at cutoff=", bestThold,
                         sep=""))
  
  # png( figFileName, width=480, height=480, units="px")   # open PNG file
  plot(perf, sub=bestMsg)
  # dev.off()
  ret <- c(bestAccuracy, bestThold)
  ret
}

#read data   
forvara = read.table("adult.data.txt",
                     sep=",",header=F,col.names=c("age", "type_employer", "fnlwgt", "education",
                                                  "education_num","marital", "occupation", "relationship", "race","sex",
                                                  "capital_gain", "capital_loss", "hr_per_week","country", "income"),
                     fill=FALSE,strip.white=T)
maena = read.table("adult.test.txt",
                   sep=",",header=F,col.names=c("age", "type_employer", "fnlwgt", "education", 
                                                "education_num","marital", "occupation", "relationship", "race","sex",
                                                "capital_gain", "capital_loss", "hr_per_week","country", "income"),
                   fill=FALSE,strip.white=T)

#combine data
totaldata <- rbind(forvara, maena)

#split data into training, testing sets
idx <- c(1:32561)
d_train <- totaldata[idx,]
d_test <- totaldata[-idx,]

###Logistic Regression Model
#--------------

#load up libraries
suppressMessages(library(ggplot2))
suppressMessages(library(caret))
suppressMessages(library(arm))
suppressMessages(library(plyr))

#eliminate non-essential parameters
d_train$capital_loss <- NULL
d_train$fnlwgt <- NULL
d_train$age <- NULL
d_train$marital <- NULL
d_train$hr_per_week <- NULL
d_train$country <- NULL

#check correctness of training data
str(d_train)                                                         #check structure of training data
names(d_train)                                                       #check headers of training data

#Set up table of all permutations
tbl1 <- gtools::combinations(8, 1)
tbl2 <- gtools::combinations(8, 2)    
tbl3 <- gtools::combinations(8, 3)
tbl4 <- gtools::combinations(8, 4)
tbl5 <- gtools::combinations(8, 5)
tbl6 <- gtools::combinations(8, 6)
tbl7 <- gtools::combinations(8, 7)
tbl8 <- gtools::combinations(8, 8)

valje <- rbind.fill.matrix(tbl1, tbl2, tbl3, tbl4, tbl5, tbl6, tbl7, tbl8)
tbl <- matrix(valje, 255, 8)
tbl[is.na(tbl)] = 0

N=255
oac <- rep(0,N)
for (i in 1:N)
{
#  d <- df[,c(i, ncol(df))]
  d <- d_train[, c(tbl[i,c(1:8)], 9)]
  m <- glm(income ~ ., data=d, family = binomial)
  prd1 <- (predict(m, newdata=d))
  prob <- 1/(1+exp(-prd1))
  bestTh <- findBestThold( prob , d$income)
  oac[i] <- bestTh[1]
}
                                                                    #accuracy: 0.8474
(mx <- max(oac))                                                    #max(oac) 
(setx <- match(mx, oac))                                            #matches max to list
tbl[setx,]                                                       
names(d_train)[tbl[setx, c(1, 2, 3, 4, 5, 6, 7, 8)]]                #ID variables of most efficient model

###randomForest Model
#--------------

#load up libraries
suppressMessages(library(randomForest))

#set up function
random_falle <- randomForest(income ~ ., data = d_train, mtry = 4, ntree = 100)
class(random_falle)                                                 #check function is correct
str(random_falle)                                                   #check structure of function
random_falle$confusion                                              #check confusion martix of function
random_falle$importance                                             #check parameter importance of function

#testing function                                                  
irja <- (predict(random_falle, newdata=d_test))                     #predicting new data with model
(tbl_rf <- table(irja, d_test$income))                              #make table with predictions, outcomes
prop.table(table(irja, d_test$income))                              #probabilities of each possibility
prop.table(table(irja, d_test$income), 1)                           #probabilities of prediction
prop.table(table(irja, d_test$income), 2)                           #probabilities of actual value
(confusionMatrix(tbl_rf))                                           #check accuracy (~0.8535)

###combination of models
#--------------

#check validity of required functions
(confusionMatrix(tbl_rf))
(confusionMatrix())