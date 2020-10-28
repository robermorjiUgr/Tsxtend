# TSxtend
## Introduction

TSxtend is a tool that will help to perform time series experimentation. The time series experimentation can be recorded with the help of the mlflow library.

It has three modules:

- ETL: It groups data [visualization algorithms](visualization.py), data splitting, to obtain different groupings and store them in csv format [partition-data](partition-data.py), as well as, algorithms to perform data mining, such as outliers elimination, [missing values](missing-values.py) and [features selections](feature_selection.py).

- MLearn: It includes algorithms that use Machine Learning techniques, such as [XGBoost](xgb.py), [Random Forest](rf_regressor.py) and [DTREE](dtre_regressor.py).

- DeepL: It includes algorithms that use Deep Learning techniques, such as [CNN](cnn.py), [LSTM](lstm.py), [MLP](mlp.py), [MLP HEADED](mlp_headed.py). 

All this is done through the execution of a simple command line. For this, it is necessary to configure a series of files, depending on the techniques to be used in our experimentation. This will do all the necessary calculations and store the results helping us to get results quickly and efficiently. 

The modules that include this tool are the next:

- ETL
    
    - [VISUALIZATION.](docs/visualization.md)
    - [PARTITION DATA.](docs/partition-data.md)
    - [MISSING VALUES.](docs/missing-values.md)
    - [FEATURES SELECTIONS.](docs/feature-selection.md)

- MLearn

  - [XGBOOST.](docs/xgb.md)
  - [RANDOM FOREST REGRESSION.](docs/rf_regression.md)
  - [DTREE REGRESSION.](docs/dtree_regression.md)

- DeepL

    - [CONVULATIONAL NETWORKS.](docs/cnn.md) 
    - [LSTM.](docs/lstm.md) 
    - [MLP.](docs/mlp.md) 
    - [MLP HEADED.](docs/mlp_headed.md) 

### Installation

### MLFLOW
~~~
pip install mlflow
~~~

### MLFLOW UI

~~~
mlflow ui
~~~

### CONDA 

