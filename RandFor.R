library(vroom)
library(tidyverse)
library(tidymodels)
library(embed)
library(doParallel)
library(themis)


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
  step_impute_median(all_numeric_predictors())

# apply the recipe to your data
prep <- prep(my_recipe)
baked <- bake(prep, new_data=test)
baked <- bake(prep, new_data = train)


my_mod <- rand_forest(mtry = tune(),
                      min_n=tune(),
                      trees=400) %>%
  set_engine("ranger") %>%
  set_mode("classification")

RandFor_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod)

## Grid of values to tune over
tuning_grid <- grid_regular(mtry(range=c(1,10)),
                            min_n(),
                            levels = 3) ## L^2 total tuning possibilities

## Split data for CV
folds <- vfold_cv(train, v = 5, repeats=1)

## Run the CV
CV_results <- RandFor_workflow %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(gain_capture)) #Or leave metrics NULL

## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best()

## Finalize the Workflow & fit it
final_wf <-
  RandFor_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data=train)


car_predictions <- predict(final_wf, new_data=test, type='prob') %>%
  rename(IsBadBuy=.pred_1) %>%
  mutate(RefId = test$RefId) %>%
  select(RefId, IsBadBuy)

vroom_write(x=car_predictions, file="./RandFOr.csv", delim=",")
stopCluster(cl)
