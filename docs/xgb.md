# XGBOOST
## file config
[xgb.yaml](../Config/xgb.yaml)

## header functions

~~~
def xgboost(file_analysis,artifact_uri,experiment_id, run_id, input_dir,n_rows,
model_input,model_output,n_splits, objective )
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
*   **objective:** [ (string) ] Params algorithms XGBRegressor

## explain use
