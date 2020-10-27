
#PYTHON
from datetime import datetime
import ipdb
import click
import yaml
import mlflow
import os

#DATASCIENCE
import pandas as pd
import numpy as np

#OWNER
import Collection.collection  as collect




@click.command(
    help="Dado un fichero CSV, transformarlo  mlflow"
)
@click.option("--date_init", type=str, default="2016-01-01", help="Fecha inicio que ayuda a discriminar por estaciones")
@click.option("--date_end",type=str, default="2016-12-31", help="Fecha final que ayuda a discriminar por estaciones")
@click.option("--path_data", type=str, default="train.csv", help="Ruta del fichero CSV")
@click.option("--n_rows", type=float, default=0, help="número de filas a extraer, 0 extrae todo")
@click.option("--fields_include", type=str, default=None, help="Incluir los siguientes campos")
@click.option("--group_by_parent", type=str, help="filtrar por un campo de las columnas")
@click.option("--output_dir", type=str,default="output/")

def PartitionDF(date_init, date_end, path_data, n_rows, 
fields_include,group_by_parent, output_dir):
    
    mlflow.set_tag("mlflow.runName", "Data Partition")
    date_init = pd.to_datetime(date_init,format="%Y-%m-%d %H:%M:%S")
    date_end  = pd.to_datetime(date_end,format="%Y-%m-%d %H:%M:%S")
    

    if not os.path.exists(output_dir+ "/partition-data"):
        os.makedirs(output_dir+ "/partition-data")  

      
    if fields_include!='None':
        fields_include = fields_include.split(",")
        df_origin = load_data(path_data, int(n_rows), fields_include)
    else:
        df_origin = load_data(path_data, int(n_rows))
        
    #import ipdb; ipdb.set_trace()
    df_origin['timestamp'] = pd.to_datetime(df_origin['timestamp'],format="%Y-%m-%d %H:%M:%S")
    mask = ( df_origin['timestamp'] >= date_init ) & ( df_origin['timestamp'] <= date_end )
    df_origin = df_origin.loc[mask]
    #df_origin[(df_origin['timestamp']>=date_init) & (df_origin['timestamp']<=date_end)]
    df_origin.set_index('timestamp',drop=True,inplace=True)

    lista_groups = []
    
    group_by_parent = group_by_parent.split(",")  
    
    for group in group_by_parent:
        lista_groups.append(df_origin[group].unique().tolist())
    
    
    query  =  []
    _query =  []      # Se almacenan las tuplas que conforman las consultas, formateadas
    _l_tupla = []   # 
    arbol = {}      # Arbol que se va formando 
    
    for ind in range(0,len(lista_groups[0])):
        query.append(createQuery(0,len(lista_groups)-1,lista_groups,arbol, lista_groups[0][ind]))
        # import ipdb; ipdb.set_trace()
    
    for item in query:
        _query.append( _format_str_query(item,group_by_parent))
        # import ipdb; ipdb.set_trace()
    
    for q_parent in _query:
        for q_child in q_parent:
            # import ipdb; ipdb.set_trace()
            df_final = df_origin.query(q_child)
            list_name_csv = _format_name_csv(q_child)
            name_csv = "/train_"
            for element in list_name_csv:
                name_csv += "_"+element[0]+"_"+element[1].replace('/',"").replace("'","")
            name_csv+=".csv"
                      
            if not df_final.empty:
                print("Creation trainning partitions: " + name_csv)
                create_csv(df_final, output_dir, name_csv,index=True)
                df_final.to_html(output_dir+ "/partition-data/"+name_csv.replace(".csv",".html")) 
            else:
                print("Not creation trainning partitions: " + name_csv + " DataFrame have not values")
    import ipdb; ipdb.set_trace()
       
    mlflow.log_artifacts(output_dir+ "/partition-data")

class Arbol:
    def __init__(self, elemento):
        self.hijos = []
        self.elemento = elemento

def agregarElemento(arbol, elemento, elementoPadre):
    subarbol = buscarSubarbol(arbol, elementoPadre);
    subarbol.hijos.append(Arbol(elemento))

def buscarSubarbol(arbol, elemento):
    if arbol.elemento == elemento:
        return arbol
    for subarbol in arbol.hijos:
        arbolBuscado = buscarSubarbol(subarbol, elemento)
        if (arbolBuscado != None):
            return arbolBuscado
    return None   

def createQuery(level,last_level,lista_groups, arbol, elementoPadre):
    _l_tupla=[]
    if level == 0 and level!=last_level:
        query=[]  
        arbol = Arbol(elementoPadre)
        _l_tupla=createQuery(level+1,last_level,lista_groups,arbol, elementoPadre)
        for item_tupla  in _l_tupla:
            tupla=[]
            for t in item_tupla:
                tupla.append( t )
            query.append(tupla)
        return query
    elif level != last_level:
        for elem in lista_groups[level]:
            agregarElemento( arbol,elem,elementoPadre )
            _l_tupla.append(createQuery(level+1,last_level, lista_groups,arbol, elem))
        return _l_tupla
    else:
        lista_tupla = []
        if level == 0:
            arbol = Arbol(elementoPadre)
            tupla = elementoPadre
            lista_tupla.append(tupla)
        else:
            for elem in lista_groups[level]:
                agregarElemento( arbol,elem,elementoPadre )
                tupla = [ elementoPadre,elem ]
                lista_tupla.append(tupla)
        # import ipdb; ipdb.set_trace()
        return lista_tupla

def _format_name_csv(elementos):
    lista_elementos = elementos.split("&")
    elementos = [ elem.replace("==","|").replace(" ","") for elem in lista_elementos ] 
    return  [ elem.split("|") for elem in elementos ]

def _format_str_query(query,group_by_parent):
    _query_format = [] 
    # import ipdb; ipdb.set_trace()
    for qs in query:
        _format_query = ""
        c = 0
        if len(group_by_parent)>1:
            for group in group_by_parent:
                if group != group_by_parent[-1]:
                    _format_query += group+"=='"+str(qs[c])+"' & "
                else:
                    _format_query += group+"=='"+str(qs[c])+"'"
                c+=1
        else:
            for group in group_by_parent:
                _format_query += group+"=='"+str(qs)+"'"
        
        _query_format.append(_format_query)
    
    return _query_format


def load_data( path, n_rows, fields=None):
       
    dataframe = collect.Collections.readCSV(path,n_rows,fields)
    return dataframe


def create_dir( path):
    collect.Collections.createDir(path)
    

def create_csv( dataframe, path, name,index=True):
    collect.Collections.createCSV(dataframe,path,name,index)

if __name__=="__main__":
    PartitionDF()

