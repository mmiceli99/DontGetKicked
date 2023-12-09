library(stacks)
library(vroom)
library(tidyverse)
library(tidymodels)
library(embed)
library(doParallel)
library(themis)
library(bonsai)
library(lightgbm)


cl <- makePSOCKcluster(10) # num_cores to use
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
  step_impute_median(all_numeric_predictors())

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

## Random Forest
rand_forest_mod <- rand_forest(mtry = tune(),
                               min_n=tune(),
                               trees = 100) %>% 
  set_engine("ranger") %>%
  set_mode("classification")

rand_forest_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(rand_forest_mod)

rand_forest_tuning_grid <- grid_regular(mtry(range = c(1, (ncol(train)-1))),
                                        min_n(),
                                        levels = 5)


# stacking
folds <- vfold_cv(train, v = 5, repeats=1)
untunedModel <- control_stack_grid()


randforest_models <- rand_forest_wf %>%
  tune_grid(resamples=folds,
            grid=rand_forest_tuning_grid,
            metrics=metric_set(gain_capture),
            control = untunedModel)

lightGBM_models <- boost_workflow %>%
  tune_grid(resamples=folds,
            grid=boost_tuning_grid,
            metrics=metric_set(gain_capture),
            control = untunedModel)


my_stack <- stacks() %>%
  add_candidates(lightGBM_models) %>%
  add_candidates(randforest_models)

stack_mod <- my_stack %>%
  blend_predictions() %>%
  fit_members()

# car_predictions <- predict(stack_mod, new_data=test, type='prob') %>%
#   rename(IsBadBuy=.pred_1) %>%
#   mutate(RefId = test$RefId) %>%
#   select(RefId, IsBadBuy)
# 
# vroom_write(x=car_predictions, file="./boost.csv", delim=",")
# stopCluster(cl)


predictions <- stack_mod %>%
  predict(new_data = test,
          type = "prob")

stacked_predictions <- predict(stack_mod, new_data = test, type='class')

submission <- predictions %>%
  mutate(RefId = test2$RefId) %>% 
  rename("IsBadBuy" = ".pred_1") %>% 
  select(3,2)

vroom_write(x = submission, file = "stacked_predictions.csv", delim=",")
