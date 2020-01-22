import pickle
import mysqlQuery_rf as rf
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

valvula = 6  # VALVULA
ruta = f'/home/arielmardones/Documentos/Respaldo-Ariel/SensoVal/Datos/val{valvula}/'

file_fecha_apertura = ruta + f'Fechas_aperturas_val{valvula}'
file_fecha_cierre = ruta + f'Fechas_cierres_val{valvula}'

with open(file_fecha_apertura, 'rb') as rf1:
    df_fechas_aperturas = pickle.load(rf1)

with open(file_fecha_cierre, 'rb') as rf2:
    df_fechas_cierre = pickle.load(rf2)

fecha_fin_apertura = df_fechas_aperturas[25][0]
df = rf.query_mysql(fecha_fin_apertura)

print('')
