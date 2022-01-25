
#PYTHON
from datetime import datetime
import ipdb
import click
import yaml
import mlflow
import os
import matplotlib.pyplot as plt
import missingno as msno
import logging
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
@click.option("--output_dir",default=None, type=str)
@click.option("--elements",default=None, type=str)

def Analysis(n_rows,fields, input_dir,output_dir,elements):
    
    mlflow.set_tag("mlflow.runName", "Data Analysis")
    #import ipdb; ipdb.set_trace();
    list_file = os.listdir(input_dir)
    list_file = [ l for l in list_file if l.endswith(".csv")]
    
    for csv in list_file:
        name_file = str(csv).replace(".csv","")
        if not os.path.exists(output_dir+ "analysis/data/"+name_file+"/"):
            os.makedirs(output_dir+ "analysis/data/"+name_file+"/bloxplot")  
            os.makedirs(output_dir+ "analysis/data/"+name_file+"/summary")  
            os.makedirs(output_dir+ "analysis/data/"+name_file+"/graph-line")  
            os.makedirs(output_dir+ "analysis/data/"+name_file+"/barplot")  
            os.makedirs(output_dir+ "analysis/data/"+name_file+"/logs")  

        
    if elements!=None:
        elements = [ elem for elem in elements.split(",") ]
        list_file = [ l for l in list_file for elem in elements if l.find(elem)!=-1 ]

    

    
    for csv in list_file:
        print("Analysis Data: " + str(csv))
        # import ipdb; ipdb.set_trace()
        name_file = str(csv).replace(".csv","")
        path = input_dir + csv
        # fields_boxplot = [ field for field in fields.split(",") ]      
        df_origin = load_data(path,n_rows)
        
        df_origin = df_origin.select_dtypes(include=['float64','int64'])
        
        ### CREATE SUMMARY HTML

        print("Create summary html")
        df_analysis = pd.DataFrame(columns=['Mean','Median','Std','Q1','Q2','Q3','Min','Max'])
        #import ipdb; ipdb.set_trace()
        df_analysis['Mean']   = df_origin.mean(axis=0,skipna=True)
        df_analysis['Median'] = df_origin.median(axis=0,skipna=True)
        df_analysis['Std']    = df_origin.std(axis=0,skipna=True)
        df_analysis['Q1']     = df_origin.quantile(q=0.25, axis=0)
        df_analysis['Q2']     = df_origin.quantile(q=0.5, axis=0)
        df_analysis['Q3']     = df_origin.quantile(q=0.75, axis=0)
        df_analysis['Min']    = df_origin.min(axis=0,skipna=True)
        df_analysis['Max']    = df_origin.max(axis=0,skipna=True)
        df_analysis.to_html(buf=output_dir+ "analysis/data/"+name_file+"/summary/summary.html", justify='justify', border='border')
        logs(df_analysis,output_dir,name_file)


        ### CREATE BOXPLOT
        # import ipdb; ipdb.set_trace()
        print("Create boxplot")
        
        data   = df_origin.values
        labels = df_origin.columns.tolist()
        #import ipdb; ipdb.set_trace();
        fig1, ax1 = plt.subplots(figsize=(15,15))
        ax1.set_title(name_file)
        ax1.set_xlabel('Fields DataSet')
        ax1.set_ylabel('Cantidad')
        ax1.boxplot(x=data,labels=labels)       

        plt.savefig(output_dir+ "analysis/data/"+name_file+"/bloxplot/boxplot.jpeg", bbox_inches='tight')
        plt.close('all')
        
        ### CREATE GRAPH LINE
        print("Create graph line")
        line = df_origin.plot.line(title=name_file)
        line.figure.savefig(output_dir+ "analysis/data/"+name_file+"/graph-line/graph_line.jpeg")
        mlflow.log_artifacts(output_dir+ "analysis")
    
def load_data( path, n_rows, fields=None):
       
    dataframe = collect.Collections.readCSV(path,n_rows,fields)
    return dataframe


def create_dir( path):
    collect.Collections.createDir(path)
    

def create_csv( dataframe, path, name):
    collect.Collections.createCSV(dataframe,path,name)

def logs (df,path,filename):
    # import ipdb; ipdb.set_trace()
    print(filename)
    FILENAME =  path + "analysis/data/"+filename+"/logs/logs.txt"
    
    f = open(FILENAME,"w+")
    
    f.write("\n\nMedia   ---- \n"        + str(df['Mean'])   )
    f.write("\n\nMediana ---- \n"        + str(df['Median']) )
    f.write("\n\nStd     ---- \n"        + str(df['Std'])    )
    f.write("\n\nQ1      ---- \n"        + str(df['Q1'])     )
    f.write("\n\nQ2      ---- \n"        + str(df['Q2'])     )
    f.write("\n\nQ3      ---- \n"        + str(df['Q3'])     )
    f.write("\n\nMin     ---- \n"        + str(df['Min'])    )
    f.write("\n\nMax     ---- \n"        + str(df['Max'])    )
    
    f.close()

if __name__=="__main__":
    Analysis()
