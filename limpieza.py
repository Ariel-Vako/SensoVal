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
dicc_n_cluster = {'201806': 0,
                  '201807': 10,
                  '201808': 20,
                  '201809': 15,
                  '201810': 20,
                  '201811': 15,
                  '201812': 20,
                  '201901': 15,
                  '201902': 15,
                  '201903': 15,
                  '201904': 15}


# Transformar a
dates = clu.from_time_to_natural(fecha)

