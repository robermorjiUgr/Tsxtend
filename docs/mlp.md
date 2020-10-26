# MLP
## file config
[mlp.yaml](../Config/mlp.yaml)

## header functions

~~~
def mlp(file_analysis,artifact_uri,experiment_id, run_id, input_dir, model_input,model_output, n_rows,
n_steps,epochs,hidden_units,batch_size,verbose)
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
*   **epochs:** [ (string) ] Epochs Neuronal Network.
*   **hidden_units:** [ (string) ] Hidden Neuronals.
*   **batch_size:** [ (string) ] Batch Size every DataSet.
*   **verbose:** [ (string) ] Verbose algorithms.


## explain use

* Config.yaml

~~~
    main.yaml
        etl:      ""
        deepl:    mlp
        mlearn:   ""
        n_rows:   0.0
        elements: ""
        output_dir: Data/test_icpe_v2

    mlp.yaml
        model_input:                air_temperature,cloud_coverage,dew_temperature,precip_depth_1_hr,sea_level_pressure,meter_reading 
        model_output:               meter_reading 
        input_dir:                  Data/test_icpe_v2
        n_steps:                    2
        hidden_units:               50
        epochs:                     10
        batch_size:                 72                 1

~~~
Este algoritmo consiste en un ml perceptron básico. El DataFrame usado por el algoritmo, será particionado en tres partes, una para el train, otro para el test y otra para la validation,actualmente están fijados en 70,10,20 respectivamente. Cabe destacar que los valores del DataFrame serán previamente normalizados.  A continuación, se realiza la secuenciación de los datos, pudiendo trocear los instervalos según el n_steps. Una vez obtenido los datos realizamos un modelo, usando en este caso una red Perceptron. Podemos ajustar los valores que nos ofrece el archivo yaml, pudiéndose añadir alguno otro más si lo deseamos. Finalmente se obtiene las métricas que vamos a medir de nuestro modelo como es (rmse, mae,r2). Se almacenará el modelo mlp, y se obtnedrá una gráfica que muestra la evolución del modelo a lo largo de la ejecución de las distintas épocas. El modelo será almacenado en el sistema con mlflow. 
