# -- coding: utf-8 --

import pickle
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scaleogram as scg
from pandas.plotting import register_matplotlib_converters
import urllib3
import os.path

from Enero2020 import mysqlQuery_rf, extracción_caract

register_matplotlib_converters()


def query_mysql(fecha_inicio, fecha_fin, válvula):
    http = urllib3.PoolManager()
    url = "http://innotech.selfip.com:8282/consulta_ssvv.php?ini='{}'&fin='{}'&id={}".format(fecha_inicio, fecha_fin, válvula)
    r = http.request('GET', url)
    resultado = list(str(r.data).split("<br>"))[:-2]

    for i in np.arange(len(resultado), 0, -1):
        if 'r' in resultado[i - 1]:
            del (resultado[i - 1])

    lista = [None] * len(resultado)

    for i, row in enumerate(resultado):
        u = row.split(',')
        lista[i] = u[1:]

    df = pd.DataFrame(lista, columns=['x', 'y', 'z'])
    df['x'] = pd.to_numeric(df.x)
    df['y'] = pd.to_numeric(df.y)
    df['z'] = pd.to_numeric(df.z)

    return df


def grafica_freq(señal, nid):
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
    ax1.set_title(f'Cierre ID {nid}')
    ax1.legend(('Eje x', 'Eje y', 'Eje z'), loc='center left',bbox_to_anchor=(1, 0.5))

    # -735
    ax2 = scg.cws(time, (señal['x'].values - señal['x'].mean()), yaxis='frequency', yscale='log', clim=(0, 100), ylim=[6, 200], cbar=None,
                  ax=ax2, cmap="jet", title='Escalograma', ylabel="x", xlabel=" ")
    # +690
    ax3 = scg.cws(time, (señal['y'].values -señal['y'].mean()), yaxis='frequency', yscale='log', cbar=None,  # clim=(0, 100), ylim=[6, 200],
                  ax=ax3, cmap="jet", ylabel="Freq [Hz]", title='', xlabel=" ")

    # +13
    ax4 = scg.cws(time, (señal['z'].values - señal['z'].mean()), yaxis='frequency', yscale='log', cbar=None,  # clim=(0, 100), ylim=[6, 200],
                  ax=ax4, cmap="jet", xlabel="Tiempo [Segundo]", ylabel="z", title='')

    # Trabajo
    directorio = '/home/arielmardones/Documentos/Respaldo-Ariel/SensoVal/Imágenes/mapaCalor-freq'
    fig.savefig(f'{directorio}/escalograma_{nid}.png', dpi=200, format='png')
    # # Home
    # directorio = r'C:\Users\ariel\OneDrive\Documentos\HStech\SensoVal\Val6\Gráficas_CWT'
    # fig.savefig(rf'{directorio}\Escalograma  {fecha.strftime("%Y-%m-%d %Hh%Mm%Ss")}.png', dpi=100, format='png')
    plt.close(fig)
    return


ruta = '/home/arielmardones/Documentos/Respaldo-Ariel/SensoVal/Datos/'
archivo = 'resumen.csv'

file = ruta + archivo

df_summary = pd.read_csv(file)

for i in range(len(df_summary)):
    fecha_inicio = df_summary['Fechas Inicio transición'][i]
    fecha_fin = df_summary['Fecha fin transición'][i]
    válvula = df_summary['Válvula'][i]
    id = df_summary['Índice'][i]

    # file_aux = ruta + 'byID/' + f'{id}.txt'
    # if os.path.exists(file_aux):
    #     with open(file_aux, 'rb') as f:
    #         df = pickle.load(f)
    # else:
    df = query_mysql(fecha_inicio, fecha_fin, válvula)

    # with open(file_aux, 'wb') as f:
    #     pickle.dump(df, f)

    grafica_freq(df, id)


