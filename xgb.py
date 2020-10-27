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
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split

#XGBOOST
import xgboost as xgb
from  xgboost import DMatrix



@click.command(help="Dado un fichero CSV, transformar en un artefacto mlflow")
@click.option("--file_analysis", type=str,default=None)
@click.option("--artifact_uri", type=str,default=None)
@click.option("--experiment_id", type=str,default=None)
@click.option("--run_id", type=str,default=None)
@click.option("--input_dir", type=str,default=None)
@click.option("--n_rows",  default=0.0,  type=float)
@click.option("--model_input",  type=str, default=None)
@click.option("--model_output",  type=str, default=None)
@click.option("--n_splits", type=int, default=10, help="Num Splits KFold")
@click.option("--objective", type=str, default="reg:squarederror", help="Objective")

def xgboost(file_analysis,artifact_uri,experiment_id, run_id, input_dir,n_rows,
model_input,model_output,n_splits, objective ):
    
    if not os.path.exists(input_dir+ "/xgboost"):
        os.makedirs(input_dir+ "/xgboost")

    
    # for file_analysis in list_file:
    print(str(file_analysis))
    mlflow.set_tag("mlflow.runName", "XGBOOST -  " + str(file_analysis.replace(".csv","").replace("train_","")))
    path = input_dir + "/"+file_analysis

    
    df_origin = load_data(path,n_rows)  

    print("XGBOOST: " + str(file_analysis))
    
    # Field Input Model
    if model_input!=None:
        df = df_origin.filter(  model_input.split(',') , axis=1)
    else:
        df = df_origin

    
    # Data Normalize
    scaler = MinMaxScaler(feature_range=(0, 1))
    df = scaler.fit_transform(df) 
    # Revisar esto para obtener los campos necesarios.
    X = df[:,0:-1]
    y = df[:,-1:]


    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    scores=[]

    for train_index, test_index in kfold.split(X):   
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        xgb_model = xgb.XGBRegressor(objective=objective)
        xgb_model.fit(X_train, y_train)
        
        y_pred = xgb_model.predict(X_test)
        
        scores.append(mean_squared_error(y_test, y_pred))

    display_scores(np.sqrt(scores))   


    
    for idx in range(len(scores)):
        mlflow.log_metric("scores",scores[idx], step=idx+1 )
    mlflow.log_metric("mean", np.mean(scores))
    mlflow.log_metric("std", np.std(scores))

    print(xgb_model.feature_importances_)
    # plot
    plt.bar(range(len(xgb_model.feature_importances_)), xgb_model.feature_importances_)
    plt.savefig(input_dir+ "/xgboost/"+file_analysis.replace(".csv",'.png')) 
    name_model = "model_xgboost_"+file_analysis.replace(".csv","")

    # SCHEMA MODEL MLFlow           
    _list_input_schema  = model_input.split(',')
    _list_output_schema = model_output.split(',')
    _list_input_schema = list ( set(_list_input_schema) - set(_list_output_schema))

    _listColSpec= [ColSpec("double",item) for item in _list_input_schema]
    input_schema = Schema(_listColSpec)
    
    _listColSpec = [ColSpec("double",item) for item in _list_output_schema]
    output_schema = Schema(_listColSpec)

    # SAVE SCHEMA MODEL
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)
    # LOG MODEL ARTIFACTS
    mlflow.xgboost.log_model(xgb_model=xgb_model,signature=signature,artifact_path=input_dir+"/xgboost" )

'''
    display_scores(): 
        - Results scores format.
'''
def display_scores(scores):
    print("Scores: {0}\nMean: {1:.3f}\nStd: {2:.3f}".format(scores, np.mean(scores), np.std(scores)))

def load_data( path, n_rows, fields=None):
    dataframe = collect.Collections.readCSV(path,n_rows,fields)
    return dataframe
   
if __name__ == '__main__':
    xgboost()
