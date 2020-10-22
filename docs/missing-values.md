# Missing Values
## file config
[missing-values.yaml](../Config/missing-values.yaml)

## header functions

~~~
def missing_values(n_rows, fields_include, input_dir,elements,alg_missing)
~~~
## parameters
*   **n_rows:**         [ (int) ] Numbers rows DataSet. This params get from [main.yaml](main.yaml)
*   **elements:**       [ (string) name_elements ] Filter by elements. This params get from [main.yaml](main.yaml)
*   **fields_include:** [ (list string) name field DataSet ] Filter by DataSet fields.
*   **input_dir:**      [ (string) name directory ] Input directory to get data.
*   **alg_missing:**    [ (string) name_algorithms] Name Algorithms missing values. [interpolate, drop]

## explain use 
