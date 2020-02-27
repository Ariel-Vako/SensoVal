# -- coding: utf-8 --

import pickle
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scaleogram as scg
from pandas.plotting import register_matplotlib_converters

from Enero2020 import mysqlQuery_rf, extracción_caract

register_matplotlib_converters()


def grafica_freq(señal, fecha):
    n = len(señal)
    time = np.arange(0, n, 1, dtype=np.int32) / 400
    # escala = np.logspace(1, 4, num=200, dtype=np.int32)
    escala = np.arange(0, 100, 0.1)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(16, 8))
    fig.subplots_adjust(hspace=0.3)
    ax1.plot(time, señal['x'].values)
    ax1.plot(time, señal['y'].values)
    ax1.plot(time, señal['z'].values)
    ax1.set_xlim(0, max(time))
    ax1.set_title(f'Señal acelerómetro a las: {fecha}')
    ax1.legend(('Eje x', 'Eje y', 'Eje z'), loc='upper right')

    ax2 = scg.cws(time, (señal['x'].values - 700), yaxis='frequency', yscale='log', clim=(0, 100), ylim=[6, 200], cbar=None,
                  ax=ax2, cmap="jet", title='Escalograma', ylabel="x", xlabel=" ")

    ax3 = scg.cws(time, (señal['y'].values - 620), yaxis='frequency', yscale='log', cbar=None,  # clim=(0, 100), ylim=[6, 200],
                  ax=ax3, cmap="jet", ylabel="Freq [Hz]", title='', xlabel=" ")

    ax4 = scg.cws(time, (señal['z'].values + 10), yaxis='frequency', yscale='log', cbar=None,  # clim=(0, 100), ylim=[6, 200],
                  ax=ax4, cmap="jet", xlabel="Tiempo [Segundo]", ylabel="z", title='')

    # # Trabajo
    # directorio = '/home/arielmardones/Documentos/Respaldo-Ariel'
    # fig.savefig(f'{directorio}/escalograma_{fecha}.png', dpi=600)
    # Home
    directorio = r'C:\Users\ariel\OneDrive\Documentos\HStech\SensoVal\Val6\Gráficas_CWT'
    fig.savefig(rf'{directorio}\Escalograma  {fecha.strftime("%Y-%m-%d %Hh%Mm%Ss")}.png', dpi=100, format='png')
    plt.close(fig)
    return


valvulas = [6, 8, 9]  # VALVULA
for valvula in valvulas:
    # # ruta Trabajo
    ruta = f'/home/arielmardones/Documentos/Respaldo-Ariel/SensoVal/Datos/val{valvula}/'
    # ruta Home
    # ruta = f'C:/Users/ariel/OneDrive/Documentos/HStech/SensoVal/Val{valvula}/'

    file_fecha_apertura = ruta + f'Fechas_aperturas_val{valvula}'
    file_fecha_cierre = ruta + f'Fechas_cierres_val{valvula}'

    with open(file_fecha_apertura, 'rb') as rf1:
        df_fechas_aperturas = pickle.load(rf1)

    with open(file_fecha_cierre, 'rb') as rf2:
        df_fechas_cierre = pickle.load(rf2)

    minutos_antes = 1  # ventana de estudio
    horizonte_temporal = datetime.strptime('2018-07-19T23:00:00', '%Y-%m-%dT%H:%M:%S')
    dict_caract = {}
    for cierre in df_fechas_cierre:
        # Extracción de datos antes del cierre
        if cierre[0].date() > horizonte_temporal.date():
            df, fecha_in = mysqlQuery_rf.query_mysql(cierre[0], minutos_antes)
            # Transformación de las señales x, y, z en frecuencia sobre bandas de frecuencia
            if not df.empty:
                # grafica_freq(df, fecha_in)
                dict_caract[fecha_in] = extracción_caract.features_from_wavelet(df)

        # Guardado en base de datos

    df = pd.DataFrame()
    # Reorganizar a dataframe
    df = pd.DataFrame.from_dict(dict_caract, orient='index')
    # Agregar columna de valvula
    df['válvula'] = valvula
    # Guardamos con pickle como Dataframe
    file_apertura = ruta + f'Precierre_val{valvula}'
    with open(file_apertura, 'wb') as fl:
        pickle.dump(df, fl)
