
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
@click.option("--elements",default=None, type=str)

def Analysis(n_rows,fields, input_dir,elements):
    
    mlflow.set_tag("mlflow.runName", "Data Analysis")
       
    if not os.path.exists(input_dir+ "/analysis"):
        os.makedirs(input_dir+ "/analysis/data/bloxplot")  
        os.makedirs(input_dir+ "/analysis/data/summary")  
        os.makedirs(input_dir+ "/analysis/data/graph-line")  
        os.makedirs(input_dir+ "/analysis/data/barplot")  
        os.makedirs(input_dir+ "/analysis/data/logs")  

        

    list_file = os.listdir(input_dir)
    list_file = [ l for l in list_file if l.endswith(".csv")]
    if elements!=None:
        elements = [ elem for elem in elements.split(",") ]
        list_file = [ l for l in list_file for elem in elements if l.find(elem)!=-1 ]

    

    
    for csv in list_file:
        print("Analysis Data: " + str(csv))
        # import ipdb; ipdb.set_trace()
        name_file = str(csv).replace(".csv","")
        path = input_dir + "/" + csv
        fields_boxplot = [ field for field in fields.split(",") ]      
        df_origin = load_data(path,n_rows)
        
        
        ### CREATE SUMMARY HTML

        print("Create summary html")
        df_analysis = pd.DataFrame(columns=['Mean','Median','Std','Q1','Q2','Q3','Min','Max'])
        #import ipdb; ipdb.set_trace()
        df_analysis['Mean']   = df_origin.mean(axis=0)
        df_analysis['Median'] = df_origin.median(axis=0)
        df_analysis['Std']    = df_origin.std(axis=0)
        df_analysis['Q1']     = df_origin.quantile(q=0.25, axis=0)
        df_analysis['Q2']     = df_origin.quantile(q=0.5, axis=0)
        df_analysis['Q3']     = df_origin.quantile(q=0.75, axis=0)
        df_analysis['Min']    = df_origin.min(axis=0)
        df_analysis['Max']    = df_origin.max(axis=0)
        df_analysis.to_html(buf=input_dir+ "/analysis/data/summary/summary_"+name_file+".html", justify='justify', border='border')
        logs(df_analysis,input_dir,name_file)


        ### CREATE BOXPLOT
        # import ipdb; ipdb.set_trace()
        print("Create boxplot")
        
        data   = df_origin[fields_boxplot].values
        labels = df_origin[fields_boxplot].columns.tolist()
        
        fig1, ax1 = plt.subplots(figsize=(15,15))
        ax1.set_title(name_file)
        ax1.set_xlabel('Fields DataSet')
        ax1.set_ylabel('Cantidad')
        ax1.boxplot(data,labels=labels)       

        plt.savefig(input_dir+ "/analysis/data/bloxplot/boxplot_"+name_file+".jpeg", bbox_inches='tight')
        plt.close('all')
        
        ### CREATE GRAPH LINE
        print("Create graph line")
        for field in fields_boxplot:
            # import ipdb; ipdb.set_trace()
            data1  = df_origin.index.values
            data2 = df_origin[[field]].values
            
            # DRAW PLOT
            fig1, ax1 = plt.subplots()
            ax1.set_title(name_file)
            ax1.set_xlabel('Fields DataSet')
            ax1.set_ylabel('Cantidad')
            print("Draw graph line column = " + field )
            plt.plot(data1,data2)
            

            if not os.path.exists(input_dir+ "/analysis/data/graph-line/"+name_file):
                os.makedirs(input_dir+ "/analysis/data/graph-line/"+name_file)

            plt.savefig(input_dir+ "/analysis/data/graph-line/"+name_file+"/"+field+".jpeg")
            plt.close('all')
        
        # import ipdb; ipdb.set_trace()
        mlflow.log_artifacts(input_dir+ "/analysis")
    
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
    LOG_FILENAME =  path + "/analysis/data/logs/"+filename+".logs"

    # Set up a specific logger with our desired output level
    my_logger = logging.getLogger('MyLogger')
    my_logger.setLevel(logging.INFO)

    # Add the log message handler to the logger
    handler = logging.handlers.RotatingFileHandler(
        LOG_FILENAME
    )
    my_logger.addHandler(handler)
    # logging.basicConfig(filename=log_file, level=logging.INFO)
    my_logger.info("\n\nMedia ---- \n"      + str(df['Mean'])    )
    my_logger.info("\n\nMediana ---- \n"    + str(df['Median'])  )
    my_logger.info("\n\nStd ---- \n"        + str(df['Std'])     )
    my_logger.info("\n\nQ1 ---- \n"         + str(df['Q1'])      )
    my_logger.info("\n\nQ2 ---- \n"         + str(df['Q2'])      )
    my_logger.info("\n\nQ3 ---- \n"         + str(df['Q3'])      )
    my_logger.info("\n\nMin ---- \n"        + str(df['Min'])     )
    my_logger.info("\n\nMax ---- \n"        + str(df['Max'])     )
    

if __name__=="__main__":
    Analysis()
