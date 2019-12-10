import MySQLdb
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use('Agg')


def grafica(eje_x, eje_y, eje_z):
    plt.close('all')
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(12, 8))
    plt.gcf().canvas.set_window_title('Energía por nivel')

    # Eje x
    fecha_x = [datetime.datetime.strftime(date, '%d-%m ~ %H:%M:%S') for date in list(eje_x['fecha'])]

    n = int(eje_x['banda frecuencia'].max())
    cx = np.zeros((n + 1, len(fecha_x)))
    fechas_unicas_x = eje_x['fecha'].unique()
    for index_x, f in enumerate(fechas_unicas_x):
        largo_nivel = len(eje_x[eje_x['fecha'] == f])
        filtrado = 1000 * eje_x[eje_x['fecha'] == f].loc[:, 'energia'].values
        for nivel in range(largo_nivel):
            cx[nivel, index_x] = filtrado[nivel]

    ax1.set_title('Energía: Eje x')
    p1 = ax1.pcolormesh(fecha_x, cx, cmap="magma")
    fig.colorbar(p1, ax=ax1)
    plt.xticks(np.arange(0, n + 1), fecha_x, cx, rotation='vertical')
    # ax2.plot(rec, 'k', label='DWT smoothing', linewidth=2)
    # ax2.legend()
    # ax2.set_title(f'Cicle {ciclo + 1}: {fecha}', fontsize=18)
    # ax2.set_ylabel('Frecuencia', fontsize=16)
    # ax2.set_xlabel('Banda de frecuencia', fontsize=16)
    # ax2.grid(b=True, which='major', color='#666666')
    # ax2.grid(b=True, which='minor', color='#999999', alpha=0.4, linestyle='--')
    # ax2.minorticks_on()
    plt.show()
    return


db = MySQLdb.connect(host='192.168.3.53', port=3306, user='ariel', passwd='hstech2018', db='SVIA_MCL')
cursor = db.cursor()
query = "SELECT fecha, eje, banda_freq, energia FROM Energy_by_band WHERE fecha < '2018-07-30' AND valvula='6';"
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
