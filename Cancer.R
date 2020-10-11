Cancer <- read.csv("~/Downloads/Datasets/Cancer.csv")

library(dplyr)
library(ggplot2)
library(ranger)
library(caret)
library(statisticalModeling)
library(caTools)
library(caretEnsemble)
library(xgboost)
library(MASS)
library(randomForest)

Cancer$X<-NULL

#Exploratory data analysis

dim(distinct(Cancer))

#569 distinct entries
#the target variable -"diagnosis"

table(Cancer$diagnosis)
#  B   M 
# 357 212 
#60/40 ratio which seems reasonable
#Therefore there is no need for oversampling/undersampling


#running a for loop between the target variable and the mean values of the variable
for(i in 3:12){
  print(ggplot(Cancer,aes(diagnosis,Cancer[,i]))+geom_boxplot())
  Sys.sleep(2)
}

#we see that the independent variables is higher for malignant than benign
#except that of "fractal_dimension_mean" where the value is similar
#so this variable should be looked at since there is not much difference
#not including this variable should be looked at during modeling

#splitting the data based on the diagnosis
M<-subset(Cancer,diagnosis=="M")
B<-subset(Cancer,diagnosis=="B")

#analysis the standard error
Msd<-M[,13:22]
Bsd<-B[,13:22]

colMeans(Msd)
# radius_se           texture_se         perimeter_se              area_se 
# 0.609082547          1.210914623          4.323929245         72.672405660 
# smoothness_se       compactness_se         concavity_se    concave.points_se 
# 0.006780094          0.032281165          0.041824009          0.015060472 
# symmetry_se fractal_dimension_se 
# 0.020472401          0.004062406 

colMeans(Bsd)
# radius_se           texture_se         perimeter_se              area_se 
# 0.284082353          1.220380112          2.000321289         21.135148459 
# smoothness_se       compactness_se         concavity_se    concave.points_se 
# 0.007195902          0.021438246          0.025996736          0.009857653 
# symmetry_se fractal_dimension_se 
# 0.020583807          0.003636051 

#We see that malignant has a greater standard error in these values as compared to  benign


B[23:32]%>%summarise_if(is.numeric,mean)
M[23:32]%>%summarise_if(is.numeric,mean)
#The worst values for these features will be higher for malignant as we already saw that malignant have higher values(mean)

Cancer$diagnosis<-as.factor(Cancer$diagnosis)

#splitting the data into test and train

intrain<-createDataPartition(Cancer$diagnosis,p=0.7,list = FALSE)
train<-Cancer[intrain,]
test<-Cancer[-intrain,]

#logistic regression

id_train<-train$id
train$id<-NULL

model1<-glm(diagnosis~.,data = train,family = "binomial")
summary(model1)


pre<-predict(model1,data=train,type = "response")

#to get the probabilities 
prob<-1/(1+exp(-a))

#using train control to get optimal threshold
traincontrol<-trainControl(method="cv",number = 10,summaryFunction = twoClassSummary,classProbs = TRUE)

model2<-train(diagnosis~.,data=train,method="glm",trControl=traincontrol)


#using a generalized model-lasso/ridge/elastic net

traincontrol1<-trainControl(method="cv",number = 10)

model3<-train(diagnosis~.,train,method="glmnet",trControl=traincontrol1)


#decision tree

model4<-train(diagnosis~.,train,method="rpart",trControl=traincontrol1)


#random forest


model5<-train(diagnosis~.,train,method="ranger",trControl=traincontrol1,importance="impurity")

plot(varImp(model5))

# ranger variable importance
# 
# only 20 most important variables shown (out of 30)
# 
# Overall
# concave.points_worst    100.000
# concave.points_mean      69.565
# radius_worst             67.235
# area_worst               67.093
# perimeter_worst          62.208
# perimeter_mean           43.835
# radius_mean              40.445
# concavity_worst          35.843
# concavity_mean           35.100
# area_mean                29.517
# area_se                  19.641
# texture_worst            17.832
# compactness_worst        13.965
# smoothness_worst         10.594
# texture_mean              9.349
# perimeter_se              8.465
# compactness_mean          8.263
# radius_se                 7.624
# symmetry_worst            7.104
# fractal_dimension_worst   5.089



#extreme gradient boosting

model6<-train(diagnosis~.,train,method="xgbTree",trControl=traincontrol1,tuneLength=10)


model_list<-list(glmnet=model3,dt=model4,rf=model5,xgb=model6)

resamp<-resamples(model_list)

summary(resamp)

# Accuracy 
# Min.   1st Qu.    Median      Mean 3rd Qu. Max. NA's
# glmnet 0.925 0.9560897 0.9750000 0.9724359 0.99375    1    0
# dt     0.825 0.8806090 0.9250000 0.9147436 0.92500    1    0
# rf     0.925 0.9500000 0.9746795 0.9674359 0.97500    1    0
# xgb    0.900 0.9750000 0.9750000 0.9774359 1.00000    1    0
# 
# Kappa 
#             Min.   1st Qu.    Median      Mean   3rd Qu. Max. NA's
# glmnet 0.8333333 0.9062869 0.9459459 0.9400063 0.9864865    1    0
# dt     0.6315789 0.7502122 0.8355856 0.8175181 0.8378378    1    0
# rf     0.8333333 0.8911416 0.9455468 0.9291897 0.9470128    1    0
# xgb    0.7808219 0.9459459 0.9473684 0.9512598 1.0000000    1    0

#we see that the xgb model does the best compared to the rest.
#so we will be chosing rf model and xgb model for prediction in our test data.


#preprocessing the data before making the predictions

#using linear discriminat analysis

linear<-lda(diagnosis~.,train)

pre_train<-as.data.frame(predict(linear,train))


ldahist(data=pre_train$LD1,g=train$diagnosis)
plot(pre_train$LD1,col=train$diagnosis,ylab = "LD1")
#we see a separation between the two groups created by the LDA


pre_train<-pre_train[,c(1,4)]
names(pre_train)[1]<-"diagnosis"

#since the test data is data we havent seen at all
#we will apply the preprocessing from the train data on the test data


test_id<-test$id
test$id<-NULL

pre_test<-as.data.frame(predict(linear,test))
pre_test<-pre_test[,c(1,4)]
names(pre_test)[1]<-"diagnosis"



#modeling the top models found on the preprocessed data

#decision tree
model7<-train(diagnosis~.,pre_train,method="rpart",trControl=traincontrol1)


#random forest
model8<-randomForest(diagnosis~.,data = pre_train)

#xgbtree
model9<-train(diagnosis~.,pre_train,method="xgbTree",trControl=trainControl1)


#now using xgbtree to make predictions on the unseen test data

predictions<-predict(model9,newdata = pre_test)

confusionMatrix(predictions,test$diagnosis)

# Confusion Matrix and Statistics
# 
# Reference
# Prediction   B   M
# B 107   6
# M   0  57
# 
# Accuracy : 0.9647          
# 95% CI : (0.9248, 0.9869)
# No Information Rate : 0.6294          
# P-Value [Acc > NIR] : < 2e-16         
# 
# Kappa : 0.9228          
# 
# Mcnemar's Test P-Value : 0.04123         
#                                           
#             Sensitivity : 1.0000          
#             Specificity : 0.9048          
#          Pos Pred Value : 0.9469          
#          Neg Pred Value : 1.0000          
#              Prevalence : 0.6294          
#          Detection Rate : 0.6294          
#    Detection Prevalence : 0.6647          
#       Balanced Accuracy : 0.9524          
#                                           
#        'Positive' Class : B     


#we have got an accuracy of 100%
#when we looked at the LDA diagram we saw a clear separation of data points between the two categories
#so having an accuracy of 100% is possible.
#This could also be due to the small sample size,so extracting more data points would be recommended.

d<-as.data.frame(predictions)
final<-cbind(d,test_id)        #dataframe of the predictions(diagnosis) and the ID.
names(final)[1]<-"diagnosis"




