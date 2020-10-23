# RANDOM FOREST REGRESSION
---
## file config
[rf_regression.yaml](../Config/rf_regression.yaml)

## header functions

~~~
def RandomForest(file_analysis,artifact_uri,experiment_id, run_id, input_dir, model_input,model_output,n_rows, 
n_estimators,criterion,max_depth,min_samples_split, min_samples_leaf, min_weight_fraction_leaf,
max_features,max_leaf_nodes, min_impurity_decrease, bootstrap,oob_score, n_jobs,
random_state, verbose, warm_start, ccp_alpha, max_samples, figure,n_splits)
~~~
## parameters
*   **file_analysis:**              File analyse. This param is generate from [main.py](../main.py)
*   **artifact_uri:**               URL artifact mlflow. This param is generate from [main.py](../main.py)
*   **experiment_id:**              Experiment id mlflow. This params is generate from [main.py](../main.py)
*   **run_id:**                     Run id mlflow. This param is generate from [main.py](../main.py)
*   **input_dir:**                  [ (string) name_directory ] Directory get Data.
*   **n_rows:**                     [ (int) ] Numbers rows DataSet. This params get from [main.yaml](main.yaml)
*   **model_input:**                [ (list string) fields ] Fields input for run algorithms.
*   **model_output:**               [ (list string) fields ] Fields output for run algorithms.
*   **n_splits:**                   [ (int) ] Number trees
*   **n_estimators:**               [ (string) ] Params algorithms RandomForestRegressor
*   **criterion:**                  [ (string) ] Params algorithms RandomForestRegressor
*   **max_depth:**                  [ (string) ] Params algorithms RandomForestRegressor
*   **min_samples_split:**          [ (string) ] Params algorithms RandomForestRegressor
*   **min_samples_leaf:**           [ (string) ] Params algorithms RandomForestRegressor
*   **min_weight_fraction_leaf:**   [ (string) ] Params algorithms RandomForestRegressor
*   **max_features:**               [ (string) ] Params algorithms RandomForestRegressor
*   **max_leaf_nodes:**             [ (string) ] Params algorithms RandomForestRegressor
*   **min_impurity_decrease:**      [ (string) ] Params algorithms RandomForestRegressor
*   **bootstrap:**                  [ (string) ] Params algorithms RandomForestRegressor
*   **oob_score:**                  [ (string) ] Params algorithms RandomForestRegressor
*   **n_jobs:**                     [ (string) ] Params algorithms RandomForestRegressor
*   **random_state:**               [ (string) ] Params algorithms RandomForestRegressor
*   **verbose:**                    [ (string) ] Params algorithms RandomForestRegressor
*   **warm_start:**                 [ (string) ] Params algorithms RandomForestRegressor
*   **ccp_alpha:**                  [ (string) ] Params algorithms RandomForestRegressor
*   **max_samples:**                [ (string) ] Params algorithms RandomForestRegressor
*   **figure:**                     [ (string) ] Params algorithms RandomForestRegressor

## explain use
