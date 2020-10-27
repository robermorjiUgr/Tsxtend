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


@click.command(help="Dado un fichero CSV, transformar en un artefacto mlflow")
@click.option("--file_analysis", type=str,default=None)
@click.option("--experiment_id", type=str,default=None)
@click.option("--run_id", type=str,default=None)
@click.option("--artifact_uri", type=str,default=None)
@click.option("--input_dir", type=str,default=None)
@click.option("--model_input", type=str,default=None)
@click.option("--model_output", type=str,default=None)
@click.option("--n_rows",  default=0.0,  type=float)
@click.option("--elements", type=str, default=None)
@click.option("--n_steps", type=int, default=3, help="n_steps redes convulacionales")
@click.option("--n_features", type=int, default=3, help="n_n_features")
@click.option("--conv_filters", type=int, default=64, help="Conv Filters")
@click.option("--conv_kernel_size", type=int, default=2, help="conv_kernel_size")
@click.option("--pool_size", type=int, default=2, help="pool_size MaxPooling1D")
@click.option("--epochs", type=int, default=10, help="Epochs")
@click.option("--hidden_units", type=int, default=50, help="Hidden Units")
@click.option("--batch_size", type=int, default=72, help="Batch Size")
@click.option("--verbose", type=int, default=1, help="Verbose")

def cnn(file_analysis,artifact_uri,experiment_id, run_id, input_dir, model_input, model_output,n_rows, elements, n_steps, n_features,
conv_filters, conv_kernel_size, pool_size,epochs,hidden_units,batch_size,verbose):
    

    if not os.path.exists(input_dir+ "/cnn"):
        os.makedirs(input_dir+ "/cnn")
   
    # for file_analysis in list_file:
    print(str(file_analysis))
    mlflow.set_tag("mlflow.runName", "CNN -  " + str(file_analysis.replace(".csv","").replace("train_","")))
   
    path = input_dir + "/"+file_analysis
    
    df_origin = load_data(path,n_rows)
   
    print("CNN: " + str(file_analysis))

    
    if model_input:
        model_input = model_input.split(",")
        df = df_origin.filter(  model_input , axis=1)
    else:
        df = df_origin
    
   
    ### Normalización de los datos
    scaler = MinMaxScaler(feature_range=(0, 1))
    df = scaler.fit_transform(df) 
    
    # Split  train, validate and test
    train,  validate, test = np.split(df,[ int( .7*len(df) ), int( .9 * len(df)) ] )
  
    # Preparation data sequences TRAIN
    X,y = preparation_data(train, n_steps, model_input)
        
    # Preparation data sequences VALIDATE
    validate_X,validate_y = preparation_data(validate, n_steps, model_input)
    
    # Preparation data sequences TEST
    test_X,test_y = preparation_data(test, n_steps, model_input)
   
    
    # X.shape = (rows,n_steps,len(model_input)-1)
    n_features = X.shape[2]
    
    # MODEL
    model = Sequential()
    model.add(Conv1D(filters=conv_filters, kernel_size=conv_kernel_size, activation='relu', input_shape=(n_steps, n_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(hidden_units, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='rmsprop', loss='mse')
    history = model.fit(X, y,
                epochs=epochs, 
                batch_size=batch_size, 
                validation_data=(validate_X, validate_y), 
                verbose=1)
    
    # PREDICT
    #import ipdb; ipdb.set_trace()
    test_predict = model.predict(test_X,verbose=0)
    (rmse, mae,r2) = eval_metrics(test_y, test_predict)
    
    name_model = "model_cnn_"+file_analysis.replace(".csv","")
    
    # SCHEMA MODEL MLFlow    
    _list_input_schema  = model_input[:-1]
    _list_output_schema = model_output.split(',')
    _list_input_schema = list ( set(_list_input_schema) - set(_list_output_schema))

    _listColSpec= [ColSpec("double",item) for item in _list_input_schema]
    input_schema = Schema(_listColSpec)
    
    _listColSpec = [ColSpec("double",item) for item in _list_output_schema]
    output_schema = Schema(_listColSpec)

    # SAVE SCHEMA MODEL
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)
    # LOG MODEL ARTIFACTS
    mlflow.keras.log_model(keras_model=model,signature=signature,artifact_path=input_dir+"/cnn")
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train loss', 'validate loss'], loc='upper left')
    plt.savefig(input_dir+"/cnn/"+file_analysis.replace(".csv","") + ".png")
     
    mlflow.log_artifact(input_dir+"/cnn/"+file_analysis.replace(".csv","")+".png")

    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2",r2)
    
    #mlflow.sklearn.log_model(model,str(output_dir)+"/cnn_model")
    
            
def load_data( path, n_rows, fields=None):
    dataframe = collect.Collections.readCSV(path,n_rows,fields)
    return dataframe

def split_sequences(sequences, n_steps):    
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)
   
def preparation_data(data, n_steps, model_input):
    in_seq=[]
    # Preparation data sequences TRAIN
    for indx in range(0,len(model_input)-1):
        in_seq.append(np.array(data[:,indx]))
    out_seq = np.array(data[:,len(model_input)-1])

    reshape_seq = []

    for in_seq_ in in_seq:
        reshape_seq.append(in_seq_.reshape(len(in_seq_),1))
    out_seq = out_seq.reshape((len(out_seq),1))
    reshape_seq.append(out_seq)

    dataset = np.hstack((reshape_seq))
    return split_sequences(dataset, n_steps)

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__=="__main__":
    cnn()
   
