# PARTITION DATA
---
## file config

[partition-data.yaml](../Config/partition-data.yaml)


## header functions

~~~
def PartitionDF(date_init, date_end, path_data, n_rows,fields_include,group_by_parent, output_dir)
~~~
## parameters

*   **date_init:**[(string) Correct Date. App transform in datatime] Date init, in order to get data from DataSet. 
*   **date_end:** [(string) Correct Date. App transform in datatime] Date end, in order to get data from DataSet.
*   **path_data:** [(string) path ] Path origin DataSet.
*   **n_rows:** [ (int) ] Number rows DataSet.
*   **fields_include:** [ (list string) name field DataSet ] Filter by DataSet fields.
*   **group_by_parent:** [ (list string) name field DataSet] Goup by DataSet levels.
*   **output_dir:** [ ( string ) name directory  ] Ouput directory, to save data.

## explain use
*   Config.yaml 

    ~~~
    main.yaml
        etl:      partition-data
        deepl:    ""
        mlearn:   ""
        n_rows:   0.0
        elements: ""
        output_dir: Data/test_icpe_v2

    partition-data.yaml
        date_init: 2016-01-01 00:00:00
        date_end:  2016-12-31 00:00:00
        fields_include: None
        path_data: train.csv
        group_by_parent: meter,site_id
        output_dir: Data/test_icpe_v2
    ~~~

Selection date init  and date end in order to that the algorithms split origin dataset. If selection fields include split only realice with this fields selections. Path_data will be the path where get DataSet split. group_by_parent is params very important, ya que, group by dataset with varios levels. 

En algunas ocasiones, it need  split DataSet use any conditions, for example, meter, site_id ... With this option could group by DataSet and then split for this fields. Example: group_by_parent:meter,site_id,building, creará una serie de DataSet donde para cada valor del padre, en este caso meter, y cada site_id y building_id. 

Sería una agrupación de tres condiciones para ir creando los distintos DataSet. Para ello se crea un árbol que por recursividad va formando las distintas consultas que posteriormente el sistema irá realizando e  irá creando los distintos CSV's. Muy importante, es que e árbol se creará de izquierda a derecha, es decir, el padre sería meter, los hijos serían los valores únicos de site_id y los nietos, en este caso, serían los valores únicos de building_id. Se debe tener en cuenta que cuanto más valores se inserten en este campo, más profundo será el árbol y más computación necesitará para obtener la solución.  

Finally the result will be save in the directory definido in the params output_dir. 

[Tree Imagen]

## Return
-   Los csv obtenidos de la partición del DataSet original. Se guardarán en output_dir
-   Transformará los csv en html para que puedan ser consultados con la interfaz de usuario de MLFlow. 

