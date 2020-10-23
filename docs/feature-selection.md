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
