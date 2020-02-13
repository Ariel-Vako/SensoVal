import pickle
from pandas.plotting import register_matplotlib_converters
import scaleogram as scg
import numpy as np
import matplotlib.pyplot as plt
from Enero2020 import mysqlQuery_rf
from datetime import timedelta, datetime

register_matplotlib_converters()


def grafica_freq(señal, fecha, eje):
    n = len(señal)
    time = np.arange(0, n, 1, dtype=np.int32) / 400
    # escala = np.logspace(1, 4, num=200, dtype=np.int32)
    escala = np.arange(0, 100, 0.1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6))
    fig.subplots_adjust(hspace=0.3)
    ax1.plot(time, señal)
    ax1.set_xlim(0, max(time))
    ax1.set_title(f'Señal acelerómetro a las: {fecha}')

    # ax2 = scg.cws(time, señal, escala, yscale='log',
    #               ax=ax2, cmap="jet", ylabel="Periodo [Segundos]", xlabel="Tiempo [segunndos/400]",
    #               title=f'Escalograma sobre {eje}')
    ax2 = scg.cws(time, señal - np.mean(señal), yaxis='frequency', yscale='log', ylim=np.logspace(1, 150, 101), clim=(0, 150), cbar='horizontal',
                  cbarkw={'aspect': 30, 'pad': 0.3, 'fraction': 0.05}, ax=ax2, cmap="jet", ylabel="Frecuencia [Hz]", xlabel="Tiempo [Segundo]",
                  title=f'Escalograma sobre {eje}')

    path = '/home/arielmardones/Documentos/Respaldo-Ariel'
    fig.savefig(f'{path}/escalograma_{fecha}: Eje {eje}.png', dpi=500)
    plt.close(fig)
    return


valvula = 6  # VALVULA
ruta = f'/home/arielmardones/Documentos/Respaldo-Ariel/SensoVal/Datos/val{valvula}/'

file_fecha_apertura = ruta + f'Fechas_aperturas_val{valvula}'
file_fecha_cierre = ruta + f'Fechas_cierres_val{valvula}'

with open(file_fecha_apertura, 'rb') as rf1:
    df_fechas_aperturas = pickle.load(rf1)

with open(file_fecha_cierre, 'rb') as rf2:
    df_fechas_cierre = pickle.load(rf2)

minutos_antes = 1  # ventana de estudio
# horizonte_temporal = datetime.strptime('2018-12-01T00:00:00', '%Y-%m-%dT%H:%M:%S')
# horizonte_temporal = datetime.strptime('2018-09-01T00:00:00', '%Y-%m-%dT%H:%M:%S')
horizonte_temporal = datetime.strptime('2018-07-19T23:00:00', '%Y-%m-%dT%H:%M:%S')
for cierre in df_fechas_cierre:
    # Extracción de datos antes del cierre
    print(cierre[0])
    if cierre[0].date() > horizonte_temporal.date():
        df, fecha_in = mysqlQuery_rf.query_mysql(cierre[0], minutos_antes)
        # Transformación de las señales x, y, z en frecuencia sobre bandas de frecuencia
        if not df.empty:
            grafica_freq(df['x'].values, fecha_in, 'x')
        # grafica_freq(df['y'].values, fecha_in, 'y')
        # grafica_freq(df['z'].values, fecha_in, 'z')
        # break
        # Caracterización de las señales en frecuencia por banda y eje
        # Guardado en base de datos

print('')
