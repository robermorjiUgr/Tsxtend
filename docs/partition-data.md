# PARTITION DATA
## file config
---
[partition-data.yaml](../Config/partition-data.yaml)


## header functions
---
~~~
def PartitionDF(date_init, date_end, path_data, n_rows,fields_include,group_by_parent, output_dir)
~~~
## parameters
---
*   **date_init:** Fecha de inicio, para obtener los datos del DataSet.
*   **date_end:** Fecha de fin, para obtener los datos del DataSet.
*   **path_data:** Ruta donde se tomarán los datos que se partirán del DataSet.
*   **n_rows:** Número de filas que se tomarán del DataSet.
*   **fields_include:** Slección de una serie de columnas del DataSet.
*   **group_by_parent:** Agrupación por niveles del DataSet. 
*   **output_dir:** Directorio de salida. 

## Use
---



