# CONVULATION NETWORK
---
## file config
[cnn.yaml](../Config/cnn.yaml)

## header functions

~~~
def cnn(file_analysis,artifact_uri,experiment_id, run_id, input_dir, model_input, model_output,n_rows,  n_steps, n_features,
conv_filters, conv_kernel_size, pool_size,epochs,hidden_units,batch_size,verbose)
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
*   **n_steps:** [ (string) ] Params Split Sequences DataSet.
*   **n_features:** [ (string) ] Params CNN.
*   **conv_filters:** [ (string) ] Params CNN.
*   **conv_kernel_size:** [ (string) ] Params CNN.
*   **pool_size:** [ (string) ] Params CNN.
*   **epochs:** [ (string) ] Epochs Neuronal Network.
*   **hidden_units:** [ (string) ] Hidden Neuronals.
*   **batch_size:** [ (string) ] Batch Size every DataSet.
*   **verbose:** [ (string) ] Verbose algorithms.


## explain use
