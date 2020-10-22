# Visualization
## file config
[visualization.yaml](../Config/visualization.yaml)

## header functions

~~~
def Visualization(n_rows,field_x, field_y, graph, measures, _resample, input_dir,elements)
~~~

## parameters
*   **n_rows:** Numbers rows DataSet. This params get from [main.yaml](main.yaml)
*   **elements:** [numbers elements ] Filter by elements. This params get from [main.yaml](main.yaml)

*   **field_x:** [name field] Field X graphs.Usually this timestamp.
*   **field_y:** [name field] Field Y graphs. 
*   **graph:** [line or missing] Line Graphs timeseries or show missing values graph.
*   **measures:** [measures] Agrupar en varios campos.
*   **resample:** [ W, M, Y ] Resamples Week(W), Month(M), Y(Year). Only show data graph.
*   **input_dir:** [ name directory ] Input directory to get data.

## example

*   Config.yaml 

    ~~~
    main.yaml
        etl:      visualization
        deepl:    ""
        mlearn:   ""
        n_rows:   0.0
        elements: ""
        output_dir: Data/test_icpe_v2

    visualization.yaml
        field_x:    timestamp
        field_y:    site_id
        graph:      line
        _resample:  W
        measures:   None
        input_dir:  Data/test_icpe_v2

    ~~~

`mlflow run . --experiment-name="test visualization"`

![example temporal serie, line graph](img/v_line.png)

