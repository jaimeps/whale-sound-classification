#Author : Abhishek Singh

#------------------------#------------------------#------------------------
#---------------------# Making Predictions with H20 functions #-------------------
#------------------------#------------------------#------------------------
#Template matching features
tmp_index  <-  read.csv("~/Dropbox/AdvancedML_Project_/Data/workspace/INDEX_TEMPLATES.csv",  header  =  T)
tmp_index  <-  as.character(tmp_index[,1]) #String of all the templates

#Calling the libraries
library(h2o)  ;  library(h2oEnsemble) 

#innitializing the set up
h2o.init(nthreads  =  -1, #All threads
         max_mem_size  =  "4G",
         min_mem_size  =  '2G') #Memory for H20 cloud
h2o.removeAll()  #Clean stat
setwd("~/Dropbox/AdvancedML_Project_/Data/workspace/")  #Chaging directory for result
#Instance : Successfully connected to http://127.0.0.1:54321/ 

#Importing the files  
df  <-  h2o.importFile(  
  path  =  normalizePath("~/Dropbox/AdvancedML_Project_/Data/workspace/trainMetrics1.csv"))

df[,  1]  <-  as.factor(df[,  1])#Treating the response variable : Factor
df[,  2]  <-  as.factor(df[,  2])#Treating the response variable : Factor

#Inducing an Index variable
tmp  <-  df[df$C1  %in%  tmp_index,  ]#Excluding the template files 
tmp  <-  tmp[,  -1] #Bad indexes
tmp  <-  as.h2o(tmp) #push it to the h20 environment

df  <-  df[!df$C1  %in%  tmp_index,  ]#Excluding the template files 
df  <-  df[,  -1] #Bad indexes

splits  <-  h2o.splitFrame(#Splitting into traning & control
  df, #The dataframe
  .8, #80% train & 20% test 
  seed   =  10004  #randomization
)

#Training, test & validation sets
supertrain  <-  df  #The Full train
train  <-  h2o.assign(splits[[1]],  "train.hex")
test  <-  h2o.assign(splits[[2]],  "test.hex")

train  <-  h2o.rbind(train,  tmp)#Adding the 25 files to train set





#------------------------#------------------------#------------------------
#---------------------# Making Predictions with Random Forests #-------------------
#------------------------#------------------------#------------------------
rf1  <-  h2o.randomForest(
                          training_frame  =  train,
                          validation_frame  =  test,
                          x  =  3:length(df) , #Predictors
                          y  =  2, #Response
                          ignore_const_cols  =  TRUE, #Remove Consts
                          model_id  =  "h2o_rf_whale", #Instance
                          balance_classes  =  TRUE, #Unbalanced data
                          max_depth  =  15,  #max depth
                          binomial_double_trees  =  TRUE,#Binary Class
                          mtries  =  80,  #mtry variables
                          stopping_metric  =  "AUC", #AUC stop
                          stopping_rounds  =  3, #Stopping criteria
                          stopping_tolerance  =  .0001, #Stopping threshold 
                          score_each_iteration  =  T,#Train & validation for each Tree
                          seed  =  100001)

#checking model performance
summary(rf1) #checking the performance
rf1@model$validation_metrics  #Validation performance
h2o.confusionMatrix(rf1) #Confusion Matrix
h2o.auc(rf1,  train  =  TRUE,  valid  =  TRUE) #AUC  0.9544400 (1st Interaction)
rf1@model$variable_importances[1:50, "variable"] #Important variables






#------------------------#------------------------#------------------------
#---------------------# Making Predictions with GBM  #-------------------
#------------------------#------------------------#------------------------

# Doing a grid search to get the best hyperparameters for GBM 
learn_rate  =  seq(.2,.6,.02) #All possible learning rates
hyper_params  <-  list(learn_rate  =  learn_rate) #Parameter tuning

model_grid  <-  h2o.grid("gbm", #classifier type
                         hyper_params  =  hyper_params, #The linear search 
                         training_frame  =  train,
                         validation_frame  =  test,
                         x  =  2:length(df), #Predictors
                         y  =  1, #Response
                         distribution  =  'bernoulli', #Binomial
                         ntrees  =  50,  #Adding trees
                         ignore_const_cols  =  TRUE, #Remove Consts
                         sample_rate  =  .8,  #Out of box error
                         col_sample_rate  =  .8, #Random Subsampling 
                         balance_classes  =  TRUE,#Binary class 
                         stopping_rounds  =  3,
                         stopping_metric  =  "AUC", #AUC being the evaluation metric
                         stopping_tolerance  =  .0001,
                         score_each_iteration  =  T)  #h20's seed

#fitting the best gbm
gbm1  <-  h2o.gbm(
                  training_frame  =  train,
                  validation_frame  =  test,
                  x  =  2:length(df), #Predictors
                  y  =  1, #Response
                  distribution  =  'bernoulli', #Classification type
                  ntrees  =  50,  #Adding trees
                  ignore_const_cols  =  TRUE, #Remove Consts
                  sample_rate  =  .8,  #Out of box error
                  col_sample_rate  =  .6, #Random Subsampling 
                  balance_classes  =  TRUE,#Binary class 
                  stopping_rounds  =  3,
                  stopping_metric  =  "AUC", #AUC being the evaluation metric
                  stopping_tolerance  =  .0001,
                  score_each_iteration  =  T, 
                  learn_rate  =  .4, #learn rate showing the best results
                  model_id  =  "h2o_gbm_whale",
                  seed  =  2000001)  #h20's seed

summary(gbm1)   #GBM performance
gbm1@model$validation_metrics   #Validation performance
h2o.confusionMatrix(gbm1)
h2o.auc(gbm1,  train  =  TRUE,  valid  =  TRUE) #AUC  
gbm1@model$variable_importances[1:40, "variable"] #Important variables


#Doing a N-fold cross validation
gbm_cv  <-  h2o.gbm(
  training_frame  =  df,
  x  =  2:length(df), #Predictors
  y  =  1, #Response
  distribution  =  'bernoulli', #Classification type
  ntrees  =  50,  #Adding trees
  ignore_const_cols  =  TRUE, #Remove Consts
  sample_rate  =  .8,  #Out of box error
  col_sample_rate  =  .8, #Random Subsampling 
  balance_classes  =  TRUE,#Binary class 
  stopping_rounds  =  3,
  stopping_metric  =  "AUC", #AUC being the evaluation metric
  stopping_tolerance  =  .0001,
  score_each_iteration  =  T, 
  learn_rate  =  .4, #learn rate showing the best results
  model_id  =  "h2o_gbm_whale",
  nfolds  =  3, #3 folds
  seed  =  2000001)  #h20's seed



#------------------------#------------------------#------------------------
#----------# Making Predictions with GBM  (With Template Matching) #----------
#------------------------#------------------------#------------------------
gbm_tm  <-  h2o.gbm(
  training_frame  =  train,
  validation_frame  =  test,
  x  =  2:151, #Predictors
  y  =  1, #Response
  distribution  =  'bernoulli', #Classification type
  ntrees  =  50,  #Adding trees
  ignore_const_cols  =  TRUE, #Remove Consts
  sample_rate  =  .8,  #Out of box error
  col_sample_rate  =  .6, #Random Subsampling 
  balance_classes  =  TRUE,#Binary class 
  stopping_rounds  =  3,
  stopping_metric  =  "AUC", #AUC being the evaluation metric
  stopping_tolerance  =  .0001,
  score_each_iteration  =  T, 
  learn_rate  =  .4, 
  model_id  =  "h2o_gbm_whale",
  seed  =  2000001)  #h20's seed

summary(gbm_tm)   #GBM performance
gbm_tm@model$validation_metrics   #Validation performance
h2o.confusionMatrix(gbm_tm)
h2o.auc(gbm_tm,  train  =  TRUE,  valid  =  TRUE) #AUC  
gbm_tm@model$variable_importances[1:40, "variable"] #Important variables







#------------------------#------------------------#------------------------
#----------# Making Predictions with GBM  (Without Template Matching) #----------
#------------------------#------------------------#------------------------
#fitting the best gbm
gbm_wtm  <-  h2o.gbm(
  training_frame  =  train,
  validation_frame  =  test,
  x  =  152:length(train), #Predictors
  y  =  1, #Response
  distribution  =  'bernoulli', #Classification type
  ntrees  =  50,  #Adding trees
  ignore_const_cols  =  TRUE, #Remove Consts
  sample_rate  =  .8,  #Out of box error
  col_sample_rate  =  .6, #Random Subsampling 
  balance_classes  =  TRUE,#Binary class 
  stopping_rounds  =  3,
  stopping_metric  =  "AUC", #AUC being the evaluation metric
  stopping_tolerance  =  .0001,
  score_each_iteration  =  T, 
  learn_rate  =  .4, 
  model_id  =  "h2o_gbm_whale",
  seed  =  2000001)  #h20's seed

summary(gbm_wtm)   #GBM performance
gbm_wtm@model$validation_metrics   #Validation performance
h2o.confusionMatrix(gbm_wtm)
h2o.auc(gbm_wtm,  train  =  TRUE,  valid  =  TRUE) #AUC  
gbm_tm@model$variable_importances[1:40, "variable"] #Important variables





#------------------------#------------------------#------------------------
#---------------------# Catching missclassification #-------------------
#------------------------#------------------------#------------------------
gbm_predictions  <-  h2o.predict( #GBM
  object  =  gbm1,
  newdata  =  df)
gbm.data  <-  as.data.frame(gbm_predictions)

#Binding with the old filer
real.data  <-  read.csv("trainmetrics.csv", header  =  T)
real.data  <-  real.data[,  1]
n.test  <-  cbind(real.data,  gbm.data)[,1:2]#Selecting the concerned columns
names(n.test)  <-  c("Actual",  "Predicted")
bad.entries  <-  n.test[n.test$Actual  !=  n.test$Predicted, ] 
write.csv(bad.entries,  file  =  "missclassifications.csv")  #Exporting to csv





#------------------------#------------------------#------------------------
#---------------------# Clearing the environment #-------------------
#------------------------#------------------------#------------------------
h2o.shutdown(prompt  =  FALSE)
detach("package:h2oEnsemble",  unload  =  TRUE)
detach("package:h2o",  unload  =  TRUE)
rm(list  =  ls())  #Clearing environment


#important features
#[1] "max_0006340"   "Index"         "maxH_0006628"  "max_0003507"   "maxH_0007582"  "max_0000118"   "maxH_0006340" 
#[8] "max_0005360"   "maxH_0003507"  "xLoc_0001566"  "maxH_0001347"  "xLoc_0000996"  "max_0006722"   "hfMax2"       
#[15] "tvTime_0008"   "tvTime_0031"   "xLocH_0000970" "xLoc_0005501"  "tvTime_0032"   "maxH_0005360"  "skewTime_0035"
#[22] "skewTime_0014" "xLoc_0003507"  "xLocH_0003507" "skewTime_0013" "skewTime_0001" "hfBwd"         "xLoc_0006628" 
#[29] "max_0000996"   "bwTime_0011"   "centTime_0011" "skewTime_0033" "xLocH_0001329" "tvTime_0046"   "maxH_0001236" 
#[36] "maxH_0000126"  "maxH_0001329"  "centTime_0012" "maxH_0000118"  "yLocH_0007582"

#h2o.exportFile(train[,1], path  =  normalizePath("~/Dropbox/AdvancedML_Project_/Data/workspace/trainfiles.csv"))
#h2o.exportFile(test[,1], path  =  normalizePath("~/Dropbox/AdvancedML_Project_/Data/workspace/testfiles.csv"))
#train  <-  h2o.rbind(train,  test_tm) #pushing th files to train
#test  <-  test[!test$C1 %in%  tmp_index,  ] #modifying the test set

file  <-  read.csv("trainMetrics1.csv",  header  =  T)
write.csv(file,  "trainMetrics1.csv")
