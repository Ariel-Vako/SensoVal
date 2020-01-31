import pickle
from pandas.plotting import register_matplotlib_converters
import scaleogram as scg
import numpy as np
from Enero2020 import mysqlQuery_rf

register_matplotlib_converters()


def grafica_freq():
    escala = np.logspace(1.0, 4.0, num=200, dtype=np.int32)
    ax=scg.cws(year, data, scales, figsize=(12,6), ylabel="Period [Years]", xlabel='Tiempo [s]', yscale='log')
    ticks = ax.set_yticks([2, 4, 8, 16, 32])
    ticks = ax.set_yticklabels([2, 4, 8, 16, 32])
    return


valvula = 6  # VALVULA
ruta = f'/home/arielmardones/Documentos/Respaldo-Ariel/SensoVal/Datos/val{valvula}/'

file_fecha_apertura = ruta + f'Fechas_aperturas_val{valvula}'
file_fecha_cierre = ruta + f'Fechas_cierres_val{valvula}'

with open(file_fecha_apertura, 'rb') as rf1:
    df_fechas_aperturas = pickle.load(rf1)

with open(file_fecha_cierre, 'rb') as rf2:
    df_fechas_cierre = pickle.load(rf2)

minutos_antes = 5
for cierre in df_fechas_cierre:
    # Extracción de datos antes del cierre
    df = mysqlQuery_rf.query_mysql(cierre[0], minutos_antes)
    # Transformación de las señales x, y, z en frecuencia sobre bandas de frecuencia

    # Caracterización de las señales en frecuencia por banda y eje
    # Guardado en base de datos

print('')
