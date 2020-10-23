#PYTHON
import os
import ipdb
import math
import click 
import json
import yaml

#DATASCIENCE
import numpy as np
import pandas as pd

#MLFLOW
import  mlflow

#OWN
import Collection.collection  as collect

#SCIPY
import scipy.stats
from scipy.sparse.linalg import expm

#SKLEARN
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import kneighbors_graph



import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb


@click.command(help="Dado un fichero CSV, transformar en un artefacto mlflow")
@click.option("--n_rows", type=float, default=0, help="número de filas a extraer, 0 extrae todo")
@click.option("--fields_include", type=str, default=None, help="Incluir los siguientes campos")
@click.option("--input_dir", type=str,default="output/")
@click.option("--elements", type=str, default=None)
@click.option("--alg_fs", type=str, default="FSMeasures", help="algoritmo seleccion de características")
def feature_selection( n_rows,fields_include,input_dir, elements,alg_fs):
    
    # Directory input dirs.
    if not os.path.exists(input_dir+"/FS/"):
        os.makedirs(input_dir+"/FS/")    
    if not os.path.exists(input_dir+ "/FS/"+alg_fs+"/"):
        os.makedirs(input_dir+ "/FS/"+alg_fs+"/")

    # Lists of Files for analise
    list_file = os.listdir(input_dir)
    list_file = [ l for l in list_file if l.endswith(".csv")]
    
    # Selected particular elements for analise
    if elements!=None:
        elements = [ elem for elem in elements.split(",") ]
        list_file = [ l for l in list_file for elem in elements if l.find(elem)!=-1 ]

  
    # Set Name Run
    mlflow.set_tag("mlflow.runName", "Feature Selection")

    for csv in list_file:
        print("Selection Features: " + str(csv))
        
        # PATH DATA
        path_data = input_dir + "/"+csv

        # Selections for fields DataSet
        if fields_include!='None':
            fields_include = fields_include.split(",")
            df_origin = load_data(path_data, int(n_rows), fields_include)
        else:
            df_origin = load_data(path_data, int(n_rows))

        
        # Algorithms Feature Selections
        if alg_fs == 'correlation':
            plt = correlation(df_origin)
            # Image .png correlations
            plt.savefig(input_dir+ "/FS/"+alg_fs+"/"+csv.replace(".csv",'')+".png")
        else:
            if alg_fs == 'FSMeasures':
                feature_data = FSMeasures(df_origin)
            # Create to_html()
            feature_data.to_html(input_dir+ "/FS/"+alg_fs+"/"+csv.replace(".csv",".html")) 
        
        # Create Artifacts mlflows
        mlflow.log_artifacts(input_dir+ "/FS")
        

def load_data(path, n_rows, fields=[]):
    dataframe = collect.Collections.readCSV(path,n_rows,fields)
    return dataframe


def create_dir( path):
    collect.Collections.createDir(path)
    

def create_csv(dataframe, path, name):
    collect.Collections.createCSV(dataframe,path,name)


def correlation(data):
    f, ax = plt.subplots(figsize=(18, 18))
    sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)
    return plt
 

    
def FSMeasures(data):
    """Calculates entropy of the passed `pd.Series`
        """
    stats = pd.DataFrame()
    
    stats["mean"] = data.mean()
    stats["Std.Dev"] = data.std()
    stats["Var"] = data.var()
    df_entropy = entropy(data)
    stats['entropy'] = df_entropy
    
    df_chi = chi(data)
    stats['chi'] = df_chi
    stats['dispersion'] = dispersion(data)
    
    print (stats)
    return stats

    
def entropy(data):
    """
    Funcion para calcular la entropy

        Parameters
        ----------
        Data : list
            elementos a los que aplicarles el calculo.

    Returns:
    -------
        entropya del conjunto de datos

    Reference:
    ---------
        .. [1] McEliece, R. (2002). The theory of information and coding (Vol. 3). Cambridge University Press.

    Examples
        --------
        >>> entropy([1, 2])
        0.92

        """
    
    entropy = {}
    for field in data.columns:
        p_data = data[field].value_counts()  # counts occurrence of each value
        entropy[field] = scipy.stats.entropy(p_data)
    df = pd.DataFrame.from_dict(entropy,orient='index')
    return df

def chi(data):
    chi_x = {}
    for field in data.columns:
        stastics, value = scipy.stats.chisquare(data[field].tolist())
        if math.isnan(stastics):
            stastics = 0
        chi_x[field] = stastics
        

    df = pd.DataFrame.from_dict(chi_x,orient='index')
    return df
  
def dispersion(data):
    
    dispersion = {}

    for field in data.columns:
        # data = data + 1  # avoid 0 division
        print(field)
        
        df = data[field]
        df[ data[field]<=0] = 1.0
        aritmeticMean = np.mean(df, axis=0)

        # geometricMean = np.power(np.prod(data[field], axis=0), 1 / data[field].shape[0])
        if aritmeticMean == 0:
            R = 0
        else:
            geometricMean = scipy.stats.hmean(df)
            R = aritmeticMean / geometricMean
        
        dispersion[field] = R
    return dispersion.values()

if __name__=="__main__":
    feature_selection()