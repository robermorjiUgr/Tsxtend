# DTREE REGRESSION
---
## file config
[dtree_regression.yaml](../Config/dtree_regression.yaml)

## header functions

~~~
def DecisionTree(file_analysis,artifact_uri,experiment_id, run_id, input_dir,model_input,model_output,n_rows, 
max_depth, criterion, splitter,min_samples_split, min_samples_leaf, min_weight_fraction_leaf,max_features, random_state,
max_leaf_nodes, figure, n_splits)
~~~
## parameters
*   **file_analysis:** File analyse. This param is generate from [main.py](../main.py)
*   **artifact_uri:** URL artifact mlflow. This param is generate from [main.py](../main.py)
*   **experiment_id:** Experiment id mlflow. This params is generate from [main.py](../main.py)
*   **run_id:** Run id mlflow. This param is generate from [main.py](../main.py)
*   **input_dir:** [ (string) name_directory ] Directory get Data.
*   **n_rows:** [ (int) ] Numbers rows DataSet. This params get from [main.yaml](main.yaml)
*   **model_input:** [ (list string) fields ] Fields input for run algorithms.
*   **model_output:** [ (list string) fields ] Fields output for run algorithms.
*   **n_splits:**  [ (int) ] Number trees
*   **max_depth:** [ (string) ] Params algorithms DecisionTreeRegressor
*   **criterion:** [ (string) ] Params algorithms DecisionTreeRegressor
*   **max_depth:** [ (string) ] Params algorithms DecisionTreeRegressor
*   **splitter:** [ (string) ] Params algorithms DecisionTreeRegressor
*   **min_samples_split:** [ (string) ] Params algorithms DecisionTreeRegressor
*   **min_samples_leaf:** [ (string) ] Params algorithms DecisionTreeRegressor
*   **min_weight_fraction_leaf:** [ (string) ] Params algorithms DecisionTreeRegressor
*   **max_features:** [ (string) ] Params algorithms DecisionTreeRegressor
*   **random_state:** [ (string) ] Params algorithms DecisionTreeRegressor
*   **max_leaf_nodes:** [ (string) ] Params algorithms DecisionTreeRegressor
*   **figure:** [ (string) ] Params algorithms DecisionTreeRegressor

## explain use
