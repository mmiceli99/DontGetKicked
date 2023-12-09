library(vroom)
library(tidyverse)
library(tidymodels)
library(embed)
library(doParallel)
library(themis)
library(bonsai)
library(lightgbm)

#parallel::detectCores() #How many cores do I have?
 cl <- makePSOCKcluster(13) # num_cores to use
 registerDoParallel(cl)
 
train <- vroom("training.csv", na = c("", "NA", "NULL", "NOT AVAIL"))
test <- vroom("test.csv", na = c("", "NA", "NULL", "NOT AVAIL"))
 
my_recipe <- recipe(IsBadBuy ~., data = train) %>%
   update_role(RefId, new_role = 'ID') %>%
   update_role_requirements("ID", bake = FALSE) %>%
   step_mutate(IsBadBuy = factor(IsBadBuy), skip = TRUE) %>%
   step_mutate(IsOnlineSale = factor(IsOnlineSale)) %>%
   step_mutate_at(all_nominal_predictors(), fn = factor) %>%
   #step_mutate_at(contains("MMR"), fn = numeric) %>%
   step_rm(contains('MMR')) %>%
   step_rm(BYRNO, WheelTypeID, VehYear, VNST, VNZIP1, PurchDate, # these variables don't seem very informative, or are repetitive
           AUCGUART, PRIMEUNIT, # these variables have a lot of missing values
           Model, SubModel, Trim) %>% # these variables have a lot of levels - could also try step_other()
   step_corr(all_numeric_predictors(), threshold = .7) %>%
   step_other(all_nominal_predictors(), threshold = .0001) %>%
   step_novel(all_nominal_predictors()) %>%
   step_unknown(all_nominal_predictors()) %>%
   step_dummy(all_nominal_predictors()) %>% #dummy encoding
   #step_lencode_mixed(all_nominal_predictors(), outcome = vars(IsBadBuy))%>% # target encoding (performed worse than dummy for light GMB)
   step_impute_median(all_numeric_predictors()) %>%
   step_normalize(all_predictors()) %>%
   step_pca(all_predictors(), threshold=.9)

# apply the recipe to your data
prep <- prep(my_recipe)
baked <- bake(prep, new_data=test)
baked <- bake(prep, new_data = train)



boost_model <- boost_tree(tree_depth=tune(),
                          trees=tune(),
                          learn_rate=tune()) %>%
  set_engine("lightgbm") %>% #or "xgboost" but lightgbm is faster
  set_mode("classification")


boost_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(boost_model)

## Grid of values to tune over
boost_tuning_grid <- grid_regular(tree_depth(),
                                  trees(),
                                  learn_rate(),
                                  levels = 5) ## L^2 total tuning possibilities


## Split data for CV
folds <- vfold_cv(train, v = 5, repeats=1)


boost_CV_results <- boost_workflow %>%
  tune_grid(resamples=folds,
            grid=boost_tuning_grid,
            metrics=metric_set(gain_capture)) #Or leave metrics NULL


boost_bestTune <- boost_CV_results %>%
  select_best()



boost_final_wf <-
  boost_workflow %>%
  finalize_workflow(boost_bestTune) %>%
  fit(data=train)


car_predictions <- predict(boost_final_wf, new_data=test, type='prob') %>%
  rename(IsBadBuy=.pred_1) %>%
  mutate(RefId = test$RefId) %>%
  select(RefId, IsBadBuy)

vroom_write(x=car_predictions, file="./boost.csv", delim=",")
stopCluster(cl)
