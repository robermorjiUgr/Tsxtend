#PYTHON
import click
import mlflow
import os
import yaml
import missingno as msno
import matplotlib.pyplot as plt

#DATASCIENCE
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype

#SKLEARN
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

#SCIPY
from scipy import stats

#OWNER
import Collection.collection  as collect


from impyute.imputation.cs import fast_knn, mice
"""
* Imputation Using ( Mean/Median) Values
* Imputation Using ( Most Frequent ) or  ( Zero/Constant ) Values
* Imputation Using k-NN
* Imputation Using Multivariate Imputation by Chained Equation (MICE)
* Stochastic regression imputation
* Extrapolation and Interpolation
* Hot-Deck imputation
"""

"""
Function for fill missing values dataset

Parameter:
train(dataframe): dataset missing values
strategy(string): strategy for fill missing values ( mean, median, most_frequences, constant )

Returns:
dataframe: dataframe

"""
@click.command(help="Dado un fichero CSV, transformar en un artefacto mlflow")
@click.option("--n_rows", type=float, default=0, help="número de filas a extraer, 0 extrae todo")
@click.option("--fields_include", type=str, default=None, help="Incluir los siguientes campos")
@click.option("--input_dir",default=None, type=str)
@click.option("--elements",default=None, type=str)
@click.option("--alg_missing", type=str, default="interpolate", help="algoritmo eliminar missing values")

def missing_values(n_rows, fields_include, input_dir,elements,alg_missing):
    # Directory input dirs.
    if not os.path.exists(input_dir+ "/missing-values"):
        os.makedirs(input_dir+ "/missing-values")  
   
    # Lists of Files for analise
    list_file = os.listdir(input_dir)
    list_file = [ l for l in list_file if l.endswith(".csv")]

    # Selected particular elements for analise
    if elements!=None:
        elements = [ elem for elem in elements.split(",") ]
        list_file = [ l for l in list_file for elem in elements if l.find(elem)!=-1 ]
    
    # Set Name Run
    mlflow.set_tag("mlflow.runName", "Missing Values")

    for csv in list_file:
        print("Missing Values: " + str(csv))
        
        # PATH DATA
        path_data = input_dir + "/" + csv
               
        # Selections for fields DataSet
        if fields_include!='None':
            fields_include = fields_include.split(",")
            df_origin = load_data(path_data, int(n_rows), fields_include)
        else:
            df_origin = load_data(path_data, int(n_rows))
        
        # Create directory algorithms missing values.
        if not os.path.exists(input_dir+ "/missing-values/"+alg_missing+"/"):
            os.makedirs(input_dir+ "/missing-values/"+alg_missing+"/")

        # Algorithms missing values
        if alg_missing == 'interpolate':
            df_final = interpolate(df_origin)
        elif alg_missing == 'drop':
            df_final = DropMissingValues(df_origin)
                
        # Fill values 0.0 values NAN
        df_final.fillna(0.0, inplace=True)

        # Create CSV
        df_final.to_csv(path_data,encoding='utf-8')
        # Create to_html()
        #df_final.to_html((input_dir+ "/missing-values/"+csv).replace(".csv",".html"))

        # Create Figure Matrix Missing Values
        msno.matrix(df_final)
        plt.savefig(input_dir+ "/missing-values/"+alg_missing+"/"+csv.replace(".csv",'')+".png") 
        # Create Artifacts mlflows
        mlflow.log_artifacts(input_dir+ "/missing-values")
        
def load_data( path, n_rows, fields=None):
    dataframe = collect.Collections.readCSV(path,n_rows,fields)
    return dataframe

def create_dir( path):
    collect.Collections.createDir(path)

def create_csv(dataframe, path, name):
    collect.Collections.createCSV(dataframe,path,name)

def draw_analize_missing_values(dataframe, file,graph):
    draw_missing.DrawMissingValues(dataframe,file,graph)
 
# Remove rows with values all NAN 
def DropMissingValues(dataframe, axis=1, how='all' ):
    df = dataframe.dropna(axis=axis, how=how)
    return df

# Interpolate missing values
def interpolate(dataframe, method='linear', limit_direction='both'):
    return dataframe.interpolate(method=method, limit_direction=limit_direction)

if __name__=="__main__":
    missing_values()