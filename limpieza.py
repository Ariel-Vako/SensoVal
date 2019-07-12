import pandas as pd
import pickle
import os.path
import maya
import clustering as clu

ruta = '/media/arielmardones/HS/SensoVal/'
query_bckup = ruta + 'query_inlfux_sensoVal.txt'
fecha_file = ruta + 'fechas.txt'

with open(query_bckup, 'rb') as rf:
    df = pickle.load(rf)

with open(fecha_file, 'rb') as rf2:
    fecha = pickle.load(rf2)

# Número de cluster por mes.
dicc_n_cluster = {'20186': 0,
                  '20187': 10,
                  '20188': 20,
                  '20189': 15,
                  '201810': 20,
                  '201811': 15,
                  '201812': 20,
                  '20191': 15,
                  '20192': 15,
                  '20193': 15,
                  '20194': 15}


# Transformar a
dates = clu.cluster_monthly(fecha, dicc_n_cluster)

