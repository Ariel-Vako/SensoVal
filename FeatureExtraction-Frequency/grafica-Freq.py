import MySQLdb
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use('Agg')


def grafica(dates, eje_x, eje_y, eje_z):
    fechas = datetime.datetime.strftime(dates, '%d-%m ~ %H:%M:%S')
    plt.close('all')
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(12, 8))
    plt.gcf().canvas.set_window_title('Energía por nivel')

    y = np.arange(7)
    ax1.set_title('Energía: Eje x')
    p1 = ax1.pcolormesh(fechas, y, cmap="magma")
    fig.colorbar(p1, ax=ax1)
    plt.xticks(c, fechas, rotation='vertical')
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
fechas = list(ejex['fecha'])
ejex.drop(columns=['fecha', 'eje'], inplace=True)

ejey = grouped_eje.get_group(1)
ejey.drop(columns=['fecha', 'eje'], inplace=True)

ejez = grouped_eje.get_group(2)
ejez.drop(columns=['fecha', 'eje'], inplace=True)

print('')
