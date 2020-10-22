#PYTHON
import click
import mlflow
import os
import yaml

#DATASCIENCE
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype

#OWNER
import Collection.collection  as collect

#SCIPY
from scipy import stats
from impyute.imputation.cs import fast_knn, mice



  
@click.command(help="Dado un fichero CSV, transformar en un artefacto mlflow")
@click.option("--n_rows", type=float, default=0, help="número de filas a extraer, 0 extrae todo")
@click.option("--fields_include", type=str, default=None, help="Incluir los siguientes campos")
@click.option("--output_dir", type=str,default="output/")
@click.option("--alg_outliers", type=str, default="z_score_method_mean", help="algoritmo eliminar outliers")
@click.option("--q1", type=float, default=0.25, help=" % eliminar outliers")
@click.option("--q3", type=float, default=0.75, help=" % eliminar outliers")
@click.option("--fields_exclude", type=str, default=None, help="Campos Excluidos outliers")
@click.option("--threshold", type=float, default=3.0, help="threshold")



def outliers(output_dir,n_rows,q1,q3,fields_exclude,fields_include,alg_outliers,threshold):
    
    if not os.path.exists(output_dir+ "/outliers-values"):
        os.makedirs(output_dir+ "/outliers-values")

    list_file = os.listdir(output_dir)
    list_file = [ l for l in list_file if l.endswith(".csv")]
    fields_include = fields_include.split(",")
    #import ipdb; ipdb.set_trace()
    for csv in list_file:
        print("Outliers Values: " + str(csv))
        path = output_dir
        path += "/"+csv       
        df_origin = load_data(path,n_rows,fields_include)
               
        if alg_outliers == 'z_score_method_mean':
            df_final =z_score_method_mean(df_origin,q1,q3, fields_exclude,fields_include)
        elif alg_outliers == 'z_score_method':
            df_final = z_score_method(df_origin, threshold, fields_exclude, fields_include)
    
        df_final.to_csv(path,encoding='utf-8')
        
        df_final.to_html((output_dir+ "/outliers-values"+name_csv).replace(".csv",".html")) 
        mlflow.log_artifacts(output_dir+ "/outliers-values")
    

def load_data(path_df, n_rows, fields=[]):
    dataframe = collect.Collections.readCSV(path_df,n_rows,fields)
    return dataframe


def create_dir( path_dir):
    collect.Collections.createDir(path_dir)
    

def create_csv(dataframe, path_csv, name_csv):
    collect.Collections.createCSV(dataframe,path_csv,name_csv)


def draw_analize_outliers_values(dataframe, file,graph,fields):
    draw_outliers.DrawOutliers(dataframe,file,graph,fields)
    # draw_bokeh.Draw(dataframe,file,graph,fields)

def z_score_method_mean(df,q1,q3,fields_exclude, fields_include ):
    
    # Get only columns necessary
    
    
    df.index.name = 'index'
    processing_include = df.filter(fields_include,axis=1)
    processing_exclude = df.filter(fields_exclude, axis=1)

    q1 = processing_include.quantile(q1)
    q3 = processing_include.quantile(q3)
    IQR = q3 - q1

    filtered = (processing_include < (q1 - 1.5 * IQR)) |(processing_include > (q3 + 1.5 * IQR))
    for name in list(processing_include.columns):
        row_index = processing_include[ filtered[name]==True ][name].index
            
        for row in row_index:
            processing_include.loc[row, name] = processing_include[name].mean()
    
    
    df = pd.merge(processing_exclude, processing_include,how='inner', on='index')
    
    
    return df


def z_score_method( df, threshold, fields_exclude, fields_include ):
    
    #Excluyo meter, id_building y timestamp
    df.index.name = 'index'
    processing_include = df.filter(fields_include, axis=1)
    processing_exclude = df.filter(fields_exclude, axis=1)
    
    processing_include[(np.abs(stats.zscore(processing_include)) < threshold).all(axis=1)]
    # Hago un merge con los dos DataFrame
    df = pd.merge(processing_exclude, processing_include,how='inner', on='index' )
    return df

"""
def gaussian 
Función que calcula la función gaussiana de un elemento

params:
mu => Media 
sigma2 = > varianza
x => valor

f(x) = 1 / sqrt( 2 * pi * sigma2 )^ e -( x - mu )^2 / 2*sigma2

return:
devuelve  el valor de la gaussiana

"""

def gaussian(mu, sigma2, x):

    coefficient = 1.0/sqrt(2.0 * pi * sigma2)
    exponential = exp ( -0.5 * ( x - mu ) ** 2 / sigma2 )
    return coefficient * exponential


def update(mean1, var1, mean2, var2):
    new_mean = ( var2*mean1 + var1*mean2) /( var2+var1 )
    new_var  = 1 / ( 1 / var2 + 1/ var1 )

    return [new_mean, new_var]


def predict( mean1, var1, mean2, var2):
    new_mean = mean1 + mean2
    new_var = var1 + var2

    return [ new_mean, new_var]

"""
def KalmanFilter( timestamp, train )
"""    

def KalmanFilter( timestamp, train ):
    
    mu  = train[0]
    sig = 10000 
    for n in range (len(timestamp)-1):
        mu, sig = Outliers.update(mu,sig, timestamp[n], timestamp[n+1])
        print ("Update => ",mu,sig )
        
        mu, sig = Outliers.predict(mu,sig,train[n], train[n+1] )
        print ("Prediction => ",mu,sig )
    return [mu, sig]

if __name__=="__main__":
    outliers()