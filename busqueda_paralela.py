import pandas as pd
import pickle
import numpy as np
from concurrent.futures import ProcessPoolExecutor


def encontrar_brechas(df, sector):

    return


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
df['fecha'].loc
print('')
