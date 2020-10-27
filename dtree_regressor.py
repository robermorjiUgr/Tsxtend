#PYTHON
import os
import ipdb
import math
import click 
import yaml
import matplotlib.pyplot as plt

#DATASCIENCE
import numpy as np
import pandas as pd



#MLFLOW
import  mlflow
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

#OWN
import Collection.collection  as collect


#KERAS
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras import layers
from keras.models import Sequential,Model
from keras.layers import Embedding, SimpleRNN,LSTM,Dense,Input, Flatten
from keras.optimizers import RMSprop
from keras.layers.merge import concatenate
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

#SKLEARN
from sklearn import tree
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

#XGBOOST
import xgboost as xgb
from  xgboost import DMatrix




@click.command(help="Dado un fichero CSV, transformar en un artefacto mlflow")
@click.option("--file_analysis", type=str,default=None)
@click.option("--artifact_uri", type=str,default=None)
@click.option("--experiment_id", type=str,default=None)
@click.option("--run_id", type=str,default=None)
@click.option("--input_dir", type=str,default=None)
@click.option("--model_input", type=str,default=None)
@click.option("--model_output", type=str,default=None)
@click.option("--n_rows",  default=0.0,  type=float)
@click.option("--max_depth", type=int, default=1, help="max_depth")
@click.option("--criterion", type=str, default='mse', help="criterion")
@click.option("--splitter", type=str, default="best", help="splitter")
@click.option("--min_samples_split", type=int, default=2, help="min_samples_split")
@click.option("--min_samples_leaf", type=int, default=1, help="min_samples_leaf")
@click.option("--min_weight_fraction_leaf", type=float, default=0., help="min_weight_fraction_leaf")
@click.option("--max_features", type=str, default="auto", help="max_features")
@click.option("--max_leaf_nodes", type=int, default=2, help="max_leaf_nodes")
@click.option("--random_state", type=int, default=0, help="random_state")
@click.option("--figure", type=str, default=False, help="figure")
@click.option("--n_splits", type=int, default=False, help="figure")

def DecisionTree(file_analysis,artifact_uri,experiment_id, run_id, input_dir,model_input,model_output,n_rows,
max_depth, criterion, splitter,min_samples_split, min_samples_leaf, min_weight_fraction_leaf,max_features, random_state,
max_leaf_nodes, figure, n_splits):
    
    if not os.path.exists(input_dir+ "/decision_tree_regressor"):
        os.makedirs(input_dir+ "/decision_tree_regressor")
 
    # for file_analysis in list_file:
    print(str(file_analysis))
    mlflow.set_tag("mlflow.runName", "DECISION TREE REGRESSOR -  " + str(file_analysis.replace(".csv","").replace("train_","")))
    path = input_dir+ "/"+file_analysis
    df_origin = load_data(path,n_rows)
    
    print("DECISION  TREE REGRESSOR: " + str(file_analysis))

    import ipdb; ipdb.set_trace()
    if model_input:
        model_input = model_input.split(',')
        df = df_origin.filter(  model_input , axis=1)
    else:
        df = df_origin
   
    dtree_model =  DecisionTreeRegressor(max_depth=max_depth, criterion=criterion, splitter=splitter,
                                     min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                     min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
                                     random_state=random_state, max_leaf_nodes=max_leaf_nodes)

    ### Normalización de los datos
    scaler = MinMaxScaler(feature_range=(0, 1))
    df = scaler.fit_transform(df) 
    # import ipdb; ipdb.set_trace()
    # Revisar esto para obtener los campos necesarios.
    X = df[:,0:len(model_input)-1]
    y = df[:,len(model_input)-1:]


    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    scores=[]

    for train_index, test_index in kfold.split(X):   
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        dtree_model.fit(X_train, y_train.ravel())
        y_pred = dtree_model.predict(X_test)
        scores.append(mean_squared_error(y_test, y_pred))
    
    display_scores(np.sqrt(scores))
   
    
    for idx in range(len(scores)):
        mlflow.log_metric("scores",scores[idx], step=idx+1 )
    mlflow.log_metric("mean", np.mean(scores))
    mlflow.log_metric("std", np.std(scores))
    
    model_output = model_output.split(',')
    tree.plot_tree(dtree_model,
                feature_names = model_input, 
                class_names= model_output,
                filled = True)
    plt.savefig(input_dir+ "/decision_tree_regressor/"+file_analysis.replace(".csv",'.png')) 
    name_model = "model_dtree_regressor_"+file_analysis.replace(".csv","")
    
    # SCHEMA MODEL MLFlow   
    _list_input_schema  = model_input
    _list_output_schema = model_output
    _list_input_schema = list ( set(_list_input_schema) - set(_list_output_schema))

    _listColSpec= [ColSpec("double",item) for item in _list_input_schema]
    input_schema = Schema(_listColSpec)
    
    _listColSpec = [ColSpec("double",item) for item in _list_output_schema]
    output_schema = Schema(_listColSpec)
    # SAVE SCHEMA MODEL
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)
    # LOG MODEL ARTIFACTS
    mlflow.sklearn.log_model(sk_model=dtree_model,signature=signature,artifact_path=input_dir+"/dtree_regressor" )

def display_scores(scores):
    print("Scores: {0}\nMean: {1:.3f}\nStd: {2:.3f}".format(scores, np.mean(scores), np.std(scores)))

def load_data( path, n_rows, fields=None):
    dataframe = collect.Collections.readCSV(path,n_rows,fields)
    return dataframe
   
if __name__ == '__main__':
    DecisionTree()
