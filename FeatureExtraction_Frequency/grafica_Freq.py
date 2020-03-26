import MySQLdb
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from os import path

matplotlib.use('Agg')


def grafica(eje_x, eje_y, eje_z):
    plt.close('all')
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(12, 8))
    plt.gcf().canvas.set_window_title('Energía por nivel')

    # Eje x
    n = int(eje_x['banda frecuencia'].max())
    fechas_unicas = eje_x['fecha'].unique()
    cx = np.zeros((len(fechas_unicas), n + 1))
    y1, x1 = np.meshgrid(np.arange(0, n + 1), np.arange(0, len(fechas_unicas)))
    for index_x, f in enumerate(fechas_unicas):
        largo_nivel = len(eje_x[eje_x['fecha'] == f])
        filtrado = 10 * eje_x[eje_x['fecha'] == f].loc[:, 'energia'].values
        for nivel in range(largo_nivel):
            cx[index_x, nivel] = min(filtrado[nivel], 100)

    ax1.set_title('Energía: Eje x')
    p1 = ax1.pcolormesh(x1, y1, cx, cmap="plasma")
    fig.colorbar(p1, ax=ax1)

    # Eje y
    n = int(eje_y['banda frecuencia'].max())
    cy = np.zeros((len(fechas_unicas), n + 1))
    y2, x2 = np.meshgrid(np.arange(0, n + 1), np.arange(0, len(fechas_unicas)))
    for index_y, f in enumerate(fechas_unicas):
        largo_nivel = len(eje_y[eje_y['fecha'] == f])
        filtrado = eje_y[eje_y['fecha'] == f].loc[:, 'energia'].values
        for nivel in range(largo_nivel):
            cy[index_y, nivel] = min(filtrado[nivel], 100)

    ax2.set_title('Energía: Eje y')
    p2 = ax2.pcolormesh(x2, y2,cy, cmap="plasma")
    fig.colorbar(p2, ax=ax2)

    # Eje z
    n = int(eje_z['banda frecuencia'].max())
    # fecha_z = [datetime.datetime.strftime(date, '%d-%m ~ %H:%M:%S') for date in list(fechas_unicas_z)]
    cz = np.zeros((len(fechas_unicas), n + 1))
    y3, x3 = np.meshgrid(np.arange(0, n + 1), np.arange(0, len(fechas_unicas)))
    for index_z, f in enumerate(fechas_unicas):
        largo_nivel = len(eje_z[eje_z['fecha'] == f])
        filtrado = eje_z[eje_z['fecha'] == f].loc[:, 'energia'].values
        for nivel in range(largo_nivel):
            cz[index_z, nivel] = min(filtrado[nivel], 100)

    ax3.set_title('Energía: Eje z')
    p3 = ax3.pcolormesh(x3, y3,cz, cmap="plasma")
    # plt.xticks(np.arange(len(fechas_unicas_z)), fechas_unicas_z, rotation='vertical')
    fig.colorbar(p3, ax=ax3)

    # plt.show()
    ruta_windows = 'C:/Users/ariel/OneDrive/Documentos/HStech/SensoVal/Datos pre-cierre/'
    if path.exists(ruta_windows):
        fig.savefig(ruta_windows + 'val6.png')
    else:
        fig.savefig('/home/arielmardones/Documentos/val6.png')

    return


if __name__ == '__main__':
    db = MySQLdb.connect(host='192.168.3.53', port=3306, user='ariel', passwd='hstech2018', db='SVIA_MCL')
    cursor = db.cursor()
    query = "SELECT fecha, eje, banda_freq, energia FROM Energy_by_band WHERE fecha < '2018-07-25' AND valvula='6';"
    cursor.execute(query)
    results = cursor.fetchall()
    db.close()

    df = pd.DataFrame(list(results), columns=('fecha', 'eje', 'banda frecuencia', 'energia'))
    grouped_eje = df.groupby('eje')
    ejex = grouped_eje.get_group(0)

    ejey = grouped_eje.get_group(1)

    ejez = grouped_eje.get_group(2)

    grafica(ejex, ejey, ejez)

print('')
