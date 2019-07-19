import pickle
import clustering as clu
import numpy as np
import grafica as grf
import influxQuery as inq
import os.path

ruta = '/media/arielmardones/HS/SensoVal/'
query_bckup = ruta + 'query_influx_sensoVal.txt'
fecha_file = ruta + 'fechas.txt'

# Cargar query en dataframe
with open(query_bckup, 'rb') as rf:
    df = pickle.load(rf)

# Cargar fechas
with open(fecha_file, 'rb') as rf2:
    fecha = pickle.load(rf2)

# Número de cluster por mes.
dicc_n_cluster = {'20186': 0,
                  '20187': 5,
                  '20188': 10,
                  '20189': 10,
                  '201810': 10,
                  '201811': 10,
                  '201812': 10,
                  '20191': 5,
                  '20192': 10,
                  '20193': 10,
                  '20194': 10}

# Transformar indices en fechas


indicesv6 = ruta + 'indices_cierre'
if not os.path.isfile(indicesv6):
    indices_en_cierre = clu.cluster_monthly(fecha, dicc_n_cluster)
    with open(indicesv6, 'wb') as f:
        pickle.dump(indices_en_cierre, f)

with open(indicesv6, 'rb') as f2:
    indices_en_cierre = pickle.load(f2)

for año_mes in indices_en_cierre.values():
    for cluster in año_mes.values():
        # Inicio y fin del período cerrado
        index_fin_cierre, index_inicio_apertura = cluster[0:3], cluster[-4:-1]
        # Dado que los cierres duran aprox. ~10sec y el muestreo es de 400Hz
        # se tiene que en cada transición existen al menos...
        """ 
            Los ángulos están cada 1sec y se reportan al cliente.
            Las coordenadas de las vibraciones se muestran 400 veces por segundo PERO
            se encuentran en una base de datos distintas.
            Trabajaré con los ÁNGULOS POR AHORA.
        """

        # Cierres
        df_cierre, fechas_cierre = inq.query_generator_cierre(fecha[index_fin_cierre[-1]])
        grf.gráfica_transición(fechas_cierre, df)

        # Aperturas
        angulo_apertura = np.array(df['Angulo'][index_inicio_apertura[0]:index_inicio_apertura[-1] + 12])
        fecha_apertura = fecha[index_inicio_apertura[0]:index_inicio_apertura[-1] + 12]
        grf.gráfica_transición(angulo_apertura, fecha_apertura)
