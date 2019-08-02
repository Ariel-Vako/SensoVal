import pandas as pd
import pickle
import numpy as np
from influxdb import InfluxDBClient
from concurrent.futures import ProcessPoolExecutor
import maya
import grafica as grf
from datetime import timedelta, datetime
import influxQuery as ifx


def mapeo_fechas(cadena):
    return maya.MayaDT.from_rfc3339(cadena).datetime()


def query_zeros(fecha_inicio, fecha_fin):
    cliente = InfluxDBClient(host='192.168.0.178', port=8086, username='', password='', database='SVALVIA_MCL')
    consulta = "SELECT angulo_sensor as Angulo, time as Fecha FROM angulos_svia WHERE angulo_sensor<= 8 and angulo_sensor>= -8 and id_sensor = '6' AND time  >= '{}' AND time<= '{}'".format(fecha_inicio.strftime("%Y-%m-%d %H:%M:%S"), fecha_fin.strftime("%Y-%m-%d %H:%M:%S"))
    resultado = cliente.query(consulta)
    cliente.close()

    df = pd.DataFrame(list(resultado.get_points()))

    e = ProcessPoolExecutor()
    fecha = list(e.map(mapeo_fechas, df['Fecha']))
    df['fecha'] = fecha
    df.pop('Fecha')
    return df


def monotonia(dataframe):
    dataframe['ángulo siguiente'] = dataframe['Angulo'].shift(periods=-1)
    dataframe.drop(dataframe.index[-1], inplace=True)
    dataframe['diferencia'] = dataframe['Angulo'] - dataframe['ángulo siguiente']

    fechas_monotonia = {}
    row = 0
    while row < len(dataframe):
        if dataframe['diferencia'].iloc[row] > 0:  # Cerrando
            fecha_inicio = dataframe['fecha'].iloc[row]
            i = row + 1
            while i < len(dataframe):
                if dataframe['diferencia'].iloc[i] > 0:
                    i += 1
            fecha_fin = dataframe['fecha'].iloc[i - 1]
            half_sec = (fecha_fin - fecha_inicio).total_seconds() / 2
            fecha_central = fecha_inicio + timedelta(seconds=half_sec)
            row = i
            flag = 0
        elif dataframe['diferencia'].iloc[row] <= 0:  # Abriendo
            fecha_inicio = dataframe['fecha'].iloc[row]
            i = row + 1
            while i < len(dataframe):
                if dataframe['diferencia'].iloc[i] <= 0:
                    i += 1
            fecha_fin = dataframe['fecha'].iloc[i - 1]
            half_sec = (fecha_fin - fecha_inicio).total_seconds() / 2
            fecha_central = fecha_inicio + timedelta(seconds=half_sec)
            row = i
            flag = 1
        fechas_monotonia[str(fecha_central)] = flag
    return fechas_monotonia


def last_query(date):
    fecha_inicio = date - timedelta(seconds=20)
    fecha_fin = date + timedelta(seconds=20)
    cliente = InfluxDBClient(host='192.168.0.178', port=8086, username='', password='', database='SVALVIA_MCL')
    consulta = "SELECT angulo_sensor as Angulo, time as Fecha FROM angulos_svia WHERE id_sensor = '6' AND time  >= '{}' AND time<= '{}'".format(fecha_inicio.strftime("%Y-%m-%d %H:%M:%S"), fecha_fin.strftime("%Y-%m-%d %H:%M:%S"))
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
intervalo_tiempo = 5  # segundos
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
    entorno_cero = query_zeros(sectores_con_transiente['inicio'].iloc[row], sectores_con_transiente['fin'].iloc[row])
    dicc_transientes = monotonia(entorno_cero)
    for key, value in dicc_transientes.items():
        df_transiente = last_query(datetime.strptime(key.split('+')[0],'%Y-%m-%d %H:%M:%S.%f'))
        grf.gráfica_transición(list(df_transiente['fecha']), df_transiente['Angulo'].values, "Pre-formateo")
        i, f = ifx.ventana(df_transiente['Angulo'].values, value)
        grf.gráfica_transición(list(df_transiente['fecha'].iloc[i:f]), df_transiente['Angulo'].iloc[i:f].values)

print('')
