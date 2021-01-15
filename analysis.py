
#PYTHON
from datetime import datetime
import ipdb
import click
import yaml
import mlflow
import os
import matplotlib.pyplot as plt
import missingno as msno

#DATASCIENCE
import pandas as pd
import numpy as np

#OWNER
import Collection.collection  as collect


  


@click.command(
    help="Give file CSV,  get mean,median,typic desviation,Q1,Q2,Q3,max and min and save in mlflow"
)
@click.option("--n_rows",default=None, type=float)
@click.option("--fields",default=None, type=str)
@click.option("--input_dir",default=None, type=str)
@click.option("--elements",default=None, type=str)

def Analysis(n_rows,fields, input_dir,elements):
    
    mlflow.set_tag("mlflow.runName", "Data Analysis")
       
    if not os.path.exists(input_dir+ "/analysis"):
        os.makedirs(input_dir+ "/analysis/data/bloxplot")  
        os.makedirs(input_dir+ "/analysis/data/summary")  
        os.makedirs(input_dir+ "/analysis/data/graph-line")  
        

    list_file = os.listdir(input_dir)
    list_file = [ l for l in list_file if l.endswith(".csv")]
    if elements!=None:
        elements = [ elem for elem in elements.split(",") ]
        list_file = [ l for l in list_file for elem in elements if l.find(elem)!=-1 ]


    # import ipdb; ipdb.set_trace()
    for csv in list_file:
        print("Analysis Data: " + str(csv))
        name_file = str(csv).replace(".csv","")
        path = input_dir
        # W_miss_value_name_png = ""
        path += "/"+csv
        # W_miss_value_name_png += 'ashrae_with_miss_value_site_id_' + str(n) + ".png"
        df_origin = load_data(path,n_rows)
        df_analysis = pd.DataFrame(columns=['Mean','Median','Std','Q1','Q2','Q3','Min','Max'])
        df_analysis['Mean']   = df_origin.mean(axis=0)
        df_analysis['Median'] = df_origin.median(axis=0)
        df_analysis['Std']    = df_origin.std(axis=0)
        df_analysis['Q1']     = df_origin.quantile(q=0.25, axis=0)
        df_analysis['Q2']     = df_origin.quantile(q=0.5, axis=0)
        df_analysis['Q3']     = df_origin.quantile(q=0.75, axis=0)
        df_analysis['Min']    = df_origin.min(axis=0)
        df_analysis['Max']    = df_origin.max(axis=0)
        print("Create summary html")

        df_analysis.to_html(buf=input_dir+ "/analysis/data/summary/summary_"+name_file+".html", justify='justify', border='border')
        print("Create boxplot")
        fields_boxplot = [ field for field in fields.split(",") ]
        boxplot = df_origin.boxplot(column=fields_boxplot,figsize=(50,50),return_type='axes')
        plt.xlabel('Fields DataSet')
        plt.ylabel('Cantidad')
        plt.savefig(input_dir+ "/analysis/data/bloxplot/boxplot_"+name_file+".jpeg")
        plt.close('all')
        print("Create graph line")
        for field in fields_boxplot:
            df_origin[[field]].plot(grid=True)

            if not os.path.exists(input_dir+ "/analysis/data/graph-line/"+name_file):
                os.makedirs(input_dir+ "/analysis/data/graph-line/"+name_file)

            plt.savefig(input_dir+ "/analysis/data/graph-line/"+name_file+"/"+field+".jpeg")
            plt.close('all')
        mlflow.log_artifacts(input_dir+ "/analysis")
    
def load_data( path, n_rows, fields=None):
       
    dataframe = collect.Collections.readCSV(path,n_rows,fields)
    return dataframe


def create_dir( path):
    collect.Collections.createDir(path)
    

def create_csv( dataframe, path, name):
    collect.Collections.createCSV(dataframe,path,name)

if __name__=="__main__":
    Analysis()