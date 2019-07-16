# Set up the work directory
setwd('C:\\Users\\Sai Krishna\\Desktop\\promotion')

#Libraries Required

library(xgboost) # for prediction
library(tidyverse) # for data wrangling
library(rBayesianOptimization) # to create cv folds and for bayesian optimisation
library(mlrMBO)  # for bayesian optimisation
library(skimr) # for summarising databases
library(DiceKriging) # mlrmbo requires this
library(rgenoud) # mlrmbo requires this
library(parallelMap)# for parallel processing
library(parallel)
library(caret)
library(rlang)
library(lubridate)# For date transformation
library(ModelMetrics)# for model metrics
library(mlr)
library(ggplot2)
library(mlbench)
library(data.table)
library(Matrix)
library(GGally)
library(ggpubr)


# Load the train data

prom <- read.csv('train.csv', stringsAsFactors = F)

# Load the test data

test <- read.csv('test.csv', stringsAsFactors = F)

# Check whether the data contain NA's or not
missing_na <- function(dataset) {
  NA_col <- apply(dataset,2,function(x)sum(is.na(x)))
  NA_col[NA_col > 0]
}

missing_na(prom)

missing_na(test)

# Check whether the data contain NA's in the form of 'spaces'
missing_space <- function(dataset){
  NA_col <- apply(dataset,2,function(x)length(which(x=='')))  
  NA_col[NA_col > 0]
}

missing_space(prom)

missing_space(test)

# Imputing missing and NA values to train 

prom$previous_year_rating[is.na(prom$previous_year_rating)] <- 6

prom$education[prom$education==''] <- 'missing'

# Imputing missing and NA values to test

test$previous_year_rating[is.na(test$previous_year_rating)] <- 6

test$education[test$education==''] <- 'missing'


# Create function to get stats about the feature

summary_col <- function(colname){
  
  unq_col   <- length(unique(colname))
  
  class_col <- class(colname)
  
  cut_col   <- if(is.numeric(colname)){table(cut(colname, breaks = 10))}
  
  output    <- list('Unique Values', unq_col, 
                    'class', class_col,
                    'cut_breaks', cut_col)
  
  return(output)                                    
}

summary_col(prom$age)

####################################
######### Correlation plot #########
####################################
correlation_plot <- function(dataset){
  ggcorr(dataset, label=TRUE, label_alpha = TRUE)  
}

correlation_plot(prom)


#######################################################
#########  Histogram of all numeric variables #########
#######################################################


hist_col <- function(dataset){
  
  dataset %>%
    keep(is.numeric) %>% 
    gather() %>% 
    ggplot(aes(value)) +
    facet_wrap(~ key, scales = "free") +
    geom_histogram()
}

hist_col(prom)

# Convert some id columns to factor class

col_names <- c('department','region','education','gender','recruitment_channel',
               'previous_year_rating','KPIs_met..80.')


prom[,col_names] <- lapply(prom[,col_names], factor)

test[,col_names] <- lapply(test[,col_names], factor)


# Finding all the categorical columns in the dataset

find_factor_cols <- function(dataset)
{
  colnames(dataset[sapply(dataset, is.factor)])
}

fact_cols <- find_factor_cols(prom)

# Bar plot for categorical variables 
bar_col <- function(dataset,target_var){
  
  for(col_num in c(1: length(fact_cols))){
    print(ggplot(dataset, aes(dataset[,fact_cols[col_num]], ..count..)) + 
            geom_bar( aes(fill = factor(dataset[,target_var])),
                      position = "dodge") +
            labs(fill = toString(fact_cols[col_num])) + 
            xlab(toString(fact_cols[col_num]))+
            theme(axis.text.x = element_text(angle = 90, hjust = 1))+
            theme(axis.text=element_text(size=16),
                  axis.title   = element_text(size=20,face="bold"),
                  legend.text  = element_text(size=16),
                  legend.title = element_text(size=20,face='bold'))+
            scale_fill_manual(values = c("limegreen", "red")))
    
  }
}

# Calling the above function 
bar_col(prom,'is_promoted')


################################################
#### Feature distribution by target variable ###
################################################

# Finding all the numeric and integer columns in the dataset

find_num_cols <- function(dataset)
{
  c(colnames(dataset[sapply(dataset, is.numeric)]),
    colnames(dataset[sapply(dataset, is.integer)]))
}


num_cols <-  unique(find_num_cols(prom))

num_cols <- num_cols[!num_cols %in%  c('employee_id','is_promoted')]

# Target variable should be converted to a factor
prom$is_promoted <- as.factor(prom$is_promoted)

# Density plot for all numeric variable w.r.t target variable

density_col <- function(dataset, target_var){
  for(num_col_name in num_cols){
    print(ggdensity(dataset, x = num_col_name,
                    add = "mean", rug = TRUE, alpha = 0.4,
                    color = target_var, fill = target_var,
                    palette = c("#0073C2FF", "#FC4E07")))
  }
}


density_col(prom,'is_promoted')  


# Class Variable 'is_promoted'

table(prom$is_promoted) #8.5 % associates got promoted among 54808 total

######################### Feature Engineering ##################################


# New Train Features
prom$work_fraction <- prom$length_of_service/ prom$age

prom$start_age <- prom$age - prom$length_of_service

prom$total_score <- prom$no_of_trainings * prom$avg_training_score


# New Test Features
test$work_fraction <- test$length_of_service/ test$age

test$start_age <- test$age - test$length_of_service

test$total_score <- test$no_of_trainings * test$avg_training_score

# Function for calculating mean encoding with smoothing

calc_smooth_mean_train <- function(df, by, on, m){
  
  # Compute the global mean
  glo_mean_val <- mean(eval(parse(text = paste0('df$',on))))
  
  # Compute the number of values and the mean of each group
  agg <- df %>%
    group_by_(by) %>%
    summarise_at(on, funs(mean,n()))
  
  loc_mean_vals  <- agg$mean
  loc_count_vals <- agg$n
  
  # Compute the "Smoothed" means
  agg$smooth = (loc_count_vals * loc_mean_vals  +  m * glo_mean_val) / 
    (loc_count_vals + m)
  
  colnames(agg)[ncol(agg)] <- paste('smooth', by, on, sep = '_')
  
  agg <- agg[c(1,ncol(agg))]
  
  df <- merge(df, agg, by=by, all.x= TRUE)
  
  return(df)
  
}

prom <- calc_smooth_mean_train(prom, 'department', 'avg_training_score', 1000)

prom <- calc_smooth_mean_train(prom, 'region', 'avg_training_score', 1000)

prom <- calc_smooth_mean_train(prom, 'education', 'avg_training_score', 1000)

prom <- calc_smooth_mean_train(prom, 'gender', 'avg_training_score', 1000)

prom <- calc_smooth_mean_train(prom, 'recruitment_channel', 'avg_training_score',1000)

prom <- calc_smooth_mean_train(prom, 'previous_year_rating','avg_training_score',1000)

prom <- calc_smooth_mean_train(prom, 'KPIs_met..80.','avg_training_score',1000)




#####################  Mean Encoding Test ######################################

# Created a new function for test set as we are using the mean value of the target
# from train and use it for test.
# Function for calculating mean encoding with smoothing and left joining the 
# same with test if there was no data for test

calc_smooth_mean_test <- function(df_train, df_test, by, on, m){
  
  # Compute the global mean
  glo_mean_val <- mean(eval(parse(text = paste0('df_train$',on))))
  
  # Compute the number of values and the mean of each group
  agg <- df_train %>%
    group_by_(by) %>%
    summarise_at(on, funs(mean,n()))
  
  loc_mean_vals  <- agg$mean
  loc_count_vals <- agg$n
  
  # Compute the "Smoothed" means
  agg$smooth = (loc_count_vals * loc_mean_vals  +  m * glo_mean_val) / 
    (loc_count_vals + m)
  
  colnames(agg)[ncol(agg)] <- paste('smooth', by, on, sep = '_')
  
  agg <- agg[c(1,ncol(agg))]
  
  df_test <- merge(df_test, agg, by=by, all.x= TRUE)
  
  return(df_test)
  
}


test <- calc_smooth_mean_test(prom, test, 'department', 'avg_training_score', 1000)

test <- calc_smooth_mean_test(prom, test, 'region', 'avg_training_score', 1000)

test <- calc_smooth_mean_test(prom, test, 'education', 'avg_training_score', 1000)

test <- calc_smooth_mean_test(prom, test, 'gender', 'avg_training_score', 1000)

test <- calc_smooth_mean_test(prom, test, 'recruitment_channel', 'avg_training_score',1000)

test <- calc_smooth_mean_test(prom, test, 'previous_year_rating','avg_training_score',1000)

test <- calc_smooth_mean_test(prom, test, 'KPIs_met..80.','avg_training_score',1000)


######################## Bayesian Optimization with mlrMBO  ####################

mydb <- prom

label_var <- "is_promoted"  

feature_vars <- mydb %>% 
  select(-one_of(c(label_var))) %>% 
  colnames()

skimr::skim(mydb ) %>% 
  skimr::kable()

# one hot encoding of categorical (factor) data
myformula <- paste0( "~", paste0( feature_vars, collapse = " + ") ) %>% 
  as.formula()

dummyFier <- caret::dummyVars(myformula, data=mydb, fullRank = TRUE)
dummyVars.df <- predict(dummyFier,newdata = mydb)
mydb_dummy <- cbind(mydb %>% select(one_of(c(label_var))), 
                    dummyVars.df)

# get  list the column names of the db with the dummy variables
feature_vars_dummy <-  mydb_dummy  %>% 
  select(-one_of(c(label_var))) %>% 
  colnames()

# Splitting train and valid data

split_index <- createDataPartition(mydb_dummy$is_promoted, p = .8, list = FALSE)

# create xgb.matrix for train
mydb_xgbmatrix_train <- xgb.DMatrix(
  data = mydb_dummy[split_index,] %>% select(feature_vars_dummy) %>% as.matrix, 
  label = mydb_dummy[split_index,] %>% pull(label_var),
  missing = NA)

# create xgb.matrix for test
mydb_xgbmatrix_valid <- xgb.DMatrix(
  data = mydb_dummy[-split_index,] %>% select(feature_vars_dummy) %>% as.matrix, 
  label = mydb_dummy[-split_index,] %>% pull(label_var),
  missing = NA)



######## xgb DMatrix creation for test data ####################################

dummyFier_test <- caret::dummyVars(myformula, data=test, fullRank = TRUE)
dummyVars.df_test <- predict(dummyFier_test, newdata = test)

# get  list the column names of the db with the dummy variables
feature_vars_dummy_test <-  dummyVars.df_test %>% colnames()

# create xgb.matrix for xgboost consumption
mydb_xgbmatrix_test <- xgb.DMatrix(
  data = dummyVars.df_test %>% as.matrix, missing = NA)



################################################################################
# random folds for xgb.cv
cv_folds = rBayesianOptimization::KFold(mydb_dummy[split_index,'is_promoted'], 
                                        nfolds= 5,
                                        stratified = TRUE,
                                        seed= 7)

# objective function: we want to maximise the log likelihood by tuning most parameters
obj.fun  <- smoof::makeSingleObjectiveFunction(
  name = "xgb_cv_bayes",
  fn =   function(x){
    set.seed(12345)
    cv <- xgb.cv(params = list(
      booster          = "gbtree",
      eta              = x["eta"],
      max_depth        = x["max_depth"],
      min_child_weight = x["min_child_weight"],
      gamma            = x["gamma"],
      subsample        = x["subsample"],
      colsample_bytree = x["colsample_bytree"],
      objective        = 'binary:logistic', 
      eval_metric      = "auc"),
      data = mydb_xgbmatrix_train,
      nround = 1000,
      folds=  cv_folds,
      prediction = FALSE,
      showsd = TRUE,
      maximize = TRUE,
      early_stopping_rounds = 50,
      verbose = 0)
    
    cv$evaluation_log[, max(test_auc_mean)]
  },
  par.set = makeParamSet(
    makeNumericParam("eta",              lower = 0.001, upper = 0.05),
    makeNumericParam("gamma",            lower = 0,     upper = 5),
    makeIntegerParam("max_depth",        lower = 1,     upper = 10),
    makeIntegerParam("min_child_weight", lower = 1,     upper = 10),
    makeNumericParam("subsample",        lower = 0.2,   upper = 1),
    makeNumericParam("colsample_bytree", lower = 0.2,   upper = 1)
  ),
  minimize = FALSE
)

# generate an optimal design with only 10  points
des = generateDesign(n=10,
                     par.set = getParamSet(obj.fun), 
                     fun = lhs::randomLHS)  ## . If no design is given by the user, mlrMBO will generate a maximin Latin Hypercube Design of size 4 times the number of the black-box function's parameters.

# bayes will have 10 additional iterations
control = makeMBOControl()
control = setMBOControlTermination(control, iters = 10)

# run this!
run = mbo(fun = obj.fun, 
          design = des,  
          control = control, 
          show.info = TRUE)

run$x

run$y


######################## XGboost CLASSIFICATION ################################

# Obtaining all the final optimization values from the parameter tuning
eta_tuning  <- run$x$eta
max_depth_tuning <- run$x$max_depth
min_child_weight_tuning <- run$x$min_child_weight
subsample_tuning <- run$x$subsample
colsample_bytree_tuning <- run$x$colsample_bytree
gamma_tuning <- run$x$gamma

params <- list(booster = 'gbtree', objective = "binary:logistic", eta=eta_tuning,
               gamma=gamma_tuning,
               max_depth=max_depth_tuning, 
               min_child_weight=min_child_weight_tuning, 
               subsample=subsample_tuning,
               colsample_bytree=colsample_bytree_tuning,
               eval_metric = 'auc')

# Cross Validation for finding the best iteration for maximum accuracy

xgbcv <- xgb.cv( params = params, data = mydb_xgbmatrix_train, nrounds = 1000,
                 nfold = 5, showsd = T, stratified = T,print_every_n = 100, 
                 early_stopping_rounds = 100, maximize = T, metrics = 'auc')


# Enter the number of rounds based on the best iteration found above

xgb1 <- xgb.train(params = params, data = mydb_xgbmatrix_train, nrounds = 1000,
                  watchlist = list(train=mydb_xgbmatrix_train, valid=mydb_xgbmatrix_valid), 
                  print_every_n = 100, early_stopping_rounds =100,
                  maximize = T,metrics = "auc")


#model prediction - valid
xgbvalid <- predict (xgb1,mydb_xgbmatrix_valid)

xgbvalid <- ifelse(xgbvalid > 0.27, 1, 0)

# F1-Score
precision <- posPredValue(as.factor(xgbvalid),
                          as.factor(mydb[-split_index,'is_promoted']),
                          positive = '1')

recall <- sensitivity(as.factor(xgbvalid),
                      as.factor(mydb[-split_index,'is_promoted']),
                      positive = '1')

F1 <- (2 * precision * recall) / (precision + recall)

F1

conf_mat_cla<-confusionMatrix(as.factor(xgbvalid),
                              as.factor(mydb[-split_index,'is_promoted']))


print(conf_mat_cla)

#model prediction - test
xgbtest <- predict (xgb1,mydb_xgbmatrix_test)

xgbtest <- ifelse(xgbtest > 0.275, 1, 0)

submission <- data.frame(employee_id = test$employee_id, 
                         is_promoted = xgbtest)

write.csv(submission, 'submission_xgboost_train_valid_mlrMBO_275.csv', row.names = FALSE)

######################### CATBOOST #############################################

set.seed(7)

split_index <- createDataPartition(prom$is_promoted, p=0.80,
                                   list=FALSE)

# select 20% of the data for validation
validation <- prom[-split_index,]

# use the remaining 80% of data to training and testing the models
training    <- prom[split_index,]

y_train <- training$is_promoted
X_train <- training %>% select(-employee_id, -is_promoted)


y_valid <- validation$is_promoted
X_valid <- validation %>% select(-employee_id, -is_promoted)

test_final <- test %>% select(-employee_id)

fit_control <- trainControl(method = "cv",
                            number = 5,
                            summaryFunction = twoClassSummary,
                            classProbs = TRUE
)

grid <- expand.grid(depth = 6, #c(6,8)
                    learning_rate = 0.01,
                    iterations = 4000,
                    l2_leaf_reg = 0.1,
                    rsm = 0.95,
                    border_count = 64)

model <- caret::train( X_train,
                       as.factor(make.names(y_train)),
                       method = catboost.caret,
                       logging_level = 'Verbose', preProc = NULL,
                       tuneGrid = grid, trControl = fit_control
)



print(model)

# Importance values for each feature
importance <- varImp(model, scale = F)

print(importance)

# Predictions
test_final <- test %>% select(-employee_id)

test_pred <- predict(model, test_final, type = "prob")

test_pred$is_promoted <- ifelse(test_pred$X1 > 0.27, 1, 0)

submission <- data.frame(employee_id = test$employee_id,
                         is_promoted =  test_pred$is_promoted)

write.csv(submission, 'catboost_newfeatures_3_27.csv', row.names = F)


######################### LGBM ################################################# 
set.seed(7)

prom$is_promoted <- as.factor(prom$is_promoted)


split_index_LGBM <- createDataPartition(prom$is_promoted, p=0.80,
                                        list=FALSE)

# select 20% of the data for validation

validation <- setDT(prom[-split_index_LGBM,])

# use the remaining 80% of data to training and testing the models

dataset    <- setDT(prom[split_index_LGBM,])

# test
setDT(test)

varnames = setdiff(colnames(data), c("employee_id", "is_promoted"))


# Sparse matrix for train, valid, & test
train_sparse = Matrix(as.matrix(dataset[, varnames, with=F]),
                      sparse=TRUE)

valid_sparse = Matrix(as.matrix(validation[, varnames, with=F]),
                      sparse=TRUE)

test_sparse  = Matrix(as.matrix(test[, varnames, with=F]),
                      sparse=TRUE)


# target values for train, valid, & test

y_train  = dataset[,is_promoted]

y_valid  = validation[,is_promoted]

test_ids = test[,employee_id]


# LGB Dataset for train, valid and test

lgb.train = lgb.Dataset(data=train_sparse, label=y_train)

lgb.valid = lgb.Dataset(data=valid_sparse, label=y_valid)

lgb.test = lgb.Dataset(data =test_sparse)

# categorical column names

col_names <- c('department','region','education','gender',
               'recruitment_channel','previous_year_rating','KPIs_met..80.')

categoricals.vec <- col_names


# Setting up LGBM parameters

lgb.grid = list(objective = "binary",
                metric = "auc",
                min_sum_hessian_in_leaf = 1,
                feature_fraction = 0.7,
                bagging_fraction = 0.7,
                bagging_freq = 5,
                min_data = 100,
                max_bin = 50,
                lambda_l1 = 8,
                lambda_l2 = 1.3,
                min_data_in_bin=100,
                min_gain_to_split = 10,
                min_data_in_leaf = 30,
                early_stopping_round = 20,
                eval_freq=20,
                is_unbalance = TRUE)

# Cross Validation

lgb.model.cv = lgb.cv(params = lgb.grid, data = lgb.train, learning_rate = 0.001, 
                      num_leaves = 20, num_threads = 2 , nrounds = 7000, 
                      early_stopping_rounds = 50,eval_freq = 20, eval = 'auc',
                      categorical_feature = categoricals.vec, nfold = 5,
                      stratified = TRUE)



# Train final model
lgb.model = lgb.train(params = lgb.grid, data = lgb.train,
                      valids = list(val = lgb.valid),learning_rate = 0.001,
                      num_leaves = 6, num_threads = 2 , nrounds = 3000,
                      eval = 'auc', categorical_feature = categoricals.vec)

# Feature Importance
tree_imp <- lgb.importance(lgb.model, percentage = TRUE)
lgb.plot.importance(tree_imp, top_n = 50, measure = "Gain")


# Predictions
test_pred_lgbm <- predict(lgb.model, test_sparse)

test_pred_lgbm <- ifelse(test_pred_lgbm > 0.27, 1, 0)

submission_lgbm = data.frame(employee_id=test_ids, 
                             is_promoted=test_pred_lgbm)

write.csv(submission_lgbm, 'submission_lgbm_27.csv', row.names = F)

############################## LGBM END ########################################