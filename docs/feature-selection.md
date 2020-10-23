# FEATURE SELECTION
---
## file config
[feature-selection.yaml](../Config/feature-selection.yaml)

## header functions

~~~
def feature_selection( n_rows,fields_include,input_dir, elements,alg_fs)
~~~
## parameters
*   **n_rows:**         [ (int) ] Numbers rows DataSet. This params get from [main.yaml](main.yaml)
*   **elements:**       [ (string) name_elements ] Filter by elements. This params get from [main.yaml](main.yaml)
*   **fields_include:** [ (list string) name field DataSet ] Filter by DataSet fields.
*   **input_dir:**      [ (string) name directory ] Input directory to get data.
*   **alg_fs:**    [ (string) name_algorithms] Name Algorithms missing values. [FSMeasure, coorelation]

## explain use 
*   Config.yaml 

    ~~~
    main.yaml
        etl:      feature-selection
        deepl:    ""
        mlearn:   ""
        n_rows:   0.0
        elements: ""
        output_dir: Data/test_icpe_v2

    missing-values.yaml
        fields_include: None
        input_dir: Data/test_icpe_v2
        alg_fs: correlation
    ~~~

It is important in this step, the fields_include parameter, because , some fields are not interesting to make, for example, an entropy calculation or to see the relation between them. Through the parameter, the algorithms will take the csv files to apply feature selections techniques.The selected algorithm will be indicated in the alg_fs parameters. Currently only FSMeasures and correlation are implemented.

- FSMeasures:
~~~
   FSMeasures show calculation. 
   *    mean. Mean.
   *    std. Desviation tipic
   *    variance
   *    entropy
   *    chi
   *    dispersion
~~~

- Correlation
~~~
    Show the correlations between fields.
~~~

[Insert example imagen with FSMeasures and Coorelations] 

## return

Save image png in:

`[input_dir]/FS/[alg_fs]`

Save html calculations FSMeasures:

`[input_dir]/FS/[alg_fs]`

Save artifacts for show in mlfow ui:

 `mlflow.log_artifacts(input_dir+ "/FS")`