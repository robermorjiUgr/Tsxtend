# TSxtend
## Introduction

TSxtend is a tool that will help to perform time series experimentation. The time series experimentation can be recorded with the help of the mlflow library.

It has three modules:

- ETL: It groups data visualization algorithms, data splitting, to obtain different groupings and store them in csv format, as well as, algorithms to perform data mining, such as outliers elimination, missing values and features extraction.

- MLearn: It includes algorithms that use Machine Learning techniques, such as XGBoost, Random Forest and DTREE.

- DeepL: It includes algorithms that use Deep Learning techniques, such as CNN, LSTM, MLP, MLP HEADED. 

All this is done through the execution of a simple command line. For this, it is necessary to configure a series of files, depending on the techniques to be used in our experimentation. This will do all the necessary calculations and store the results helping us to get results quickly and efficiently. 

The modules that include this tool are the next:

- ETL
    
    - [VISUALIZATION.](docs/visualization.md)
    - [PARTITION DATA.](docs/partition-data.md)
    - [MISSING VALUES.](docs/partition-data.md)
    - [FEATURES SELECTIONS.](docs/partition-data.md)

- MLearn

  - [XGBOOST.](docs/xgb.md)
  - [RANDOM FOREST REGRESSION.](docs/partition-data.md)
  - [DTREE REGRESSION.](docs/partition-data.md)

- DeepL

    - [CONVULATIONAL NETWORKS.](docs/xgb.md) 
    - [LSTM.](docs/xgb.md) 
    - [MLP.](docs/xgb.md) 
    - [MLP HEADED.](docs/xgb.md) 


