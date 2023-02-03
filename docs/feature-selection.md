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

    feature-selection.yaml
        fields_include: meter_reading,air_temperature,dew_temperature,precip_depth_1_hr,sea_level_pressure,wind_speed
        input_dir: Data/test_icpe_v2
        alg_fs: FSMeasures
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

### CORRELATION 
![example temporal serie, line graph](img/correlation.png)

### FSMASURES
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>Std.Dev</th>
      <th>Var</th>
      <th>entropy</th>
      <th>chi</th>
      <th>dispersion</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>meter_reading</th>
      <td>226.25</td>
      <td>376.54</td>
      <td>141,788.92</td>
      <td>6.44</td>
      <td>5.67*10^8</td>
      <td>89.44</td>
    </tr>
    <tr>
      <th>air_temperature</th>
      <td>22.86</td>
      <td>6.01</td>
      <td>36.11</td>
      <td>3.76</td>
      <td>1.43*10^6</td>
      <td>1.12</td>
    </tr>
    <tr>
      <th>dew_temperature</th>
      <td>16.85</td>
      <td>6.48</td>
      <td>42.01</td>
      <td>3.64</td>
      <td>2.25*10^6</td>
      <td>1.81</td>
    </tr>
    <tr>
      <th>precip_depth_1_hr</th>
      <td>1.38</td>
      <td>12.97</td>
      <td>168.24</td>
      <td>0.52</td>
      <td>1.10*10^8</td>
      <td>2.29</td>
    </tr>
    <tr>
      <th>sea_level_pressure</th>
      <td>1,017.95</td>
      <td>4.03</td>
      <td>16.24</td>
      <td>5.15</td>
      <td>1.44*10^4</td>
      <td>1.03</td>
    </tr>
    <tr>
      <th>wind_speed</th>
      <td>3.37</td>
      <td>2.15</td>
      <td>4.64</td>
      <td>2.59</td>
      <td>1.24*10^6</td>
      <td>1.42</td>
    </tr>
  </tbody>
</table>
## Return

Save image png in:

`[input_dir]/FS/[alg_fs]`

Save html calculations FSMeasures:

`[input_dir]/FS/[alg_fs]`

Save artifacts for show in mlfow ui:

 `mlflow.log_artifacts(input_dir+ "/FS")`
