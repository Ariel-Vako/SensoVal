import pandas as pd
import pickle
import numpy as np
from influxdb import InfluxDBClient
from concurrent.futures import ProcessPoolExecutor
import maya


def mapeo_fechas(cadena):
    return maya.MayaDT.from_rfc3339(cadena).datetime()


def query_zeros(fecha_inicio, fecha_fin):
    cliente = InfluxDBClient(host='192.168.0.178', port=8086, username='', password='', database='SVALVIA_MCL')
    consulta = "SELECT angulo_sensor as Angulo, time as Fecha FROM angulos_svia WHERE angulo_sensor<= 2 and angulo_sensor>= -2 and id_sensor = '6' AND time  >= '{}' AND time<= '{}'".format(fecha_inicio.strftime("%Y-%m-%d %H:%M:%S"), fecha_fin.strftime("%Y-%m-%d %H:%M:%S"))
    resultado = cliente.query(consulta)
    cliente.close()

    df = pd.DataFrame(list(resultado.get_points()))

    e = ProcessPoolExecutor()
    fecha = list(e.map(mapeo_fechas, df['Fecha']))
    df['fecha'] = fecha
    df.pop('Fecha')
    return df


ruta = '/media/arielmardones/HS/SensoVal/'
query_bckup = ruta + 'query_influx_sensoVal.txt'

with open(query_bckup, 'rb') as rf:
    df = pickle.load(rf)

# Eliminar mes de prueba
df.drop(df.index[0:7], inplace=True)
# Reinicio de índice
df.reset_index(inplace=True)
df.pop('index')

# Generar nuevo dataframe que contenga muestra de fechas cada "n" tiempo
segmentación = pd.DataFrame(columns=['inicio', 'fin', 'diferencia'])
intervalo_tiempo = 50  # segundos
segmentación['inicio'] = df['fecha'][0:-1:intervalo_tiempo]
segmentación['fin'] = segmentación['inicio'].shift(periods=-1)
segmentación.reset_index(inplace=True)
segmentación.drop(segmentación.index[-1], inplace=True)

# Diferencia de fechas en segundos
segmentación['diferencia'] = segmentación['fin'] - segmentación['inicio']
segmentación['diferencia'] = segmentación['diferencia'] / np.timedelta64(1, 's')

# Sectores
sectores_con_transiente = segmentación.loc[segmentación['diferencia'] > intervalo_tiempo + 15]

for row in range(len(sectores_con_transiente)):
    angulos_ceros = query_zeros(sectores_con_transiente['fecha_inicio'].iloc[row], sectores_con_transiente['fecha_fin'].iloc[row])



print('')
