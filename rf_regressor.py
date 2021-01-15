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

#XGBOOST
import xgboost as xgb
from  xgboost import DMatrix


def display_scores(scores):
    print("Scores: {0}\nMean: {1:.3f}\nStd: {2:.3f}".format(scores, np.mean(scores), np.std(scores)))

@click.command(help="Dado un fichero CSV, transformar en un artefacto mlflow")
@click.option("--file_analysis", type=str,default=None)
@click.option("--artifact_uri", type=str,default=None)
@click.option("--experiment_id", type=str,default=None)
@click.option("--run_id", type=str,default=None)
@click.option("--input_dir", type=str,default=None)
@click.option("--model_input", type=str,default=None)
@click.option("--model_output", type=str,default=None)
@click.option("--n_rows",  default=0.0,  type=float)
@click.option("--n_estimators", type=int, default=100, help="n_estimators")
@click.option("--criterion", type=str, default='mse', help="criterion")
@click.option("--max_depth", type=int, default=1, help="max_depth")
@click.option("--min_samples_split", type=int, default=2, help="min_samples_split")
@click.option("--min_samples_leaf", type=int, default=1, help="min_samples_leaf")
@click.option("--min_weight_fraction_leaf", type=float, default=0., help="min_weight_fraction_leaf")
@click.option("--max_features", type=str, default="auto", help="max_features")
@click.option("--max_leaf_nodes", type=int, default=2, help="max_leaf_nodes")
@click.option("--min_impurity_decrease", type=float, default=0., help="min_impurity_decrease")
@click.option("--bootstrap", type=str, default=True, help="bootstrap")
@click.option("--oob_score", type=str, default=False, help="oob_score")
@click.option("--n_jobs", type=int, default=-1, help="n_jobs")
@click.option("--random_state", type=int, default=0, help="random_state")
@click.option("--verbose", type=int, default=0, help="verbose")
@click.option("--warm_start", type=str, default=False, help="warm_start")
@click.option("--ccp_alpha", type=float, default=0.0, help="ccp_alpha")
@click.option("--max_samples", type=str, default=None, help="max_samples")
@click.option("--figure", type=str, default=False, help="figure")
@click.option("--n_splits", type=int, default=10, help="Num Splits KFold")

def RandomForest(file_analysis,artifact_uri,experiment_id, run_id, input_dir, model_input,model_output,n_rows, 
n_estimators,criterion,max_depth,min_samples_split, min_samples_leaf, min_weight_fraction_leaf,
max_features,max_leaf_nodes, min_impurity_decrease, bootstrap,oob_score, n_jobs,
random_state, verbose, warm_start, ccp_alpha, max_samples, figure,n_splits):


    if not os.path.exists(input_dir+ "/rf_regressor"):
        os.makedirs(input_dir+ "/rf_regressor")
      
    # for file_analysis in list_file:
    print(str(file_analysis))
    mlflow.set_tag("mlflow.runName", "RANDOM FOREST REGRESSOR -  " + str(file_analysis.replace(".csv","").replace("train_","")))
    path = input_dir + "/"+file_analysis
    
    df_origin = load_data(path,n_rows)
    

    print("RANDOM FOREST REGRESSOR: " + str(file_analysis))

    if model_input:
        model_input = model_input.split(',')
        df = df_origin.filter(  model_input , axis=1)
    else:
        df = df_origin
    
    rfr_model = RandomForestRegressor(n_estimators=n_estimators, criterion=criterion,
                                    max_depth=max_depth, min_samples_split=min_samples_split,
                                    min_samples_leaf=min_samples_leaf,
                                    min_weight_fraction_leaf=min_weight_fraction_leaf,
                                    max_features=max_features, max_leaf_nodes=max_leaf_nodes,
                                    min_impurity_decrease=min_impurity_decrease,
                                    bootstrap=bootstrap, oob_score=oob_score,
                                    n_jobs=n_jobs, random_state=random_state,
                                    verbose=verbose, max_samples=None, warm_start=warm_start,
                                    ccp_alpha=ccp_alpha)

    ### Data Normalice
    scaler = MinMaxScaler(feature_range=(0, 1))
    df = scaler.fit_transform(df) 
    
    # Split  train, validate and test
    # train,  validate, test = np.split(df,[ int( .7*len(df) ), int( .9 * len(df)) ] )
    
    # X: All columns except last columns.
    # y: last column.
    X = df[:,0:len(model_input)-1]
    y = df[:,len(model_input)-1:]


    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    scores=[]

    for train_index, test_index in kfold.split(X):   
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]        
        rfr_model.fit(X_train, y_train.ravel())        
        y_pred = rfr_model.predict(X_test)       
        scores.append(mean_squared_error(y_test, y_pred))
    
    display_scores(np.sqrt(scores))
   
    
    for idx in range(len(scores)):
        mlflow.log_metric("scores",scores[idx], step=idx+1 )
    mlflow.log_metric("mean", np.mean(scores))
    mlflow.log_metric("std", np.std(scores))
    
    fn=model_input
    cn=model_output
    # fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
    # tree.plot_tree(rfr_model.estimators_[0],
    #             feature_names = fn, 
    #             class_names=cn,
    #             filled = True);
    # plot
    plt.title("Random Forest Regression: KFold(n_split="+str(n_splits)+")")
    plt.bar(range(len(scores)), scores)
    plt.savefig(input_dir+ "/rf_regressor/"+file_analysis.replace(".csv",'.png')) 
    name_model = "model_rf_regressor_"+file_analysis.replace(".csv","")
    
    # SCHEMA MODEL MLFlow               
    _list_input_schema  = model_input
    _list_output_schema = model_output.split(',')
    _list_input_schema = list ( set(_list_input_schema) - set(_list_output_schema))

    _listColSpec= [ColSpec("double",item) for item in _list_input_schema]
    input_schema = Schema(_listColSpec)
    
    _listColSpec = [ColSpec("double",item) for item in _list_output_schema]
    output_schema = Schema(_listColSpec)
    
    # SAVE SCHEMA MODEL
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)
    # LOG MODEL ARTIFACTS
    mlflow.sklearn.log_model(sk_model=rfr_model,signature=signature,artifact_path=input_dir+"/rf_regressor" )
        

def load_data( path, n_rows, fields=None):
    dataframe = collect.Collections.readCSV(path,n_rows,fields)
    return dataframe
   
if __name__ == '__main__':
    RandomForest()
