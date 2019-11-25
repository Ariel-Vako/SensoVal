from __future__ import division
import funciones_freq as fx
import params_freq
import sys, argparse, csv, os, MySQLdb, time
from datetime import datetime, timedelta
import struct
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

id_reg = 0
for valv in [6, 8, 9]:
    db = MySQLdb.connect(host='192.168.3.53', port=3306, user='ariel', passwd='hstech2018', db='SVIA_MCL')
    cursor = db.cursor()
    if id_reg == 0:
        query = "SELECT id_reg, time, cast( data_sensor AS CHAR) FROM svia_data WHERE id_sensor = '{}' LIMIT 1;".format(valv)
    else:
        query = "SELECT id_reg, time, cast( data_sensor AS CHAR) FROM svia_data WHERE id_sensor = '{}' AND id_reg > '{}' LIMIT 1;".format(valv, id_reg)

    cursor.execute(query)
    db.close()
    results = cursor.fetchall()
    df = pd.DataFrame(list(results), columns=('id', 'fecha', 'blob'))
    id_reg = df['id'].values[0]

    cont = 0
    index = 0
    for ind, row in enumerate(results):
        df_aux = pd.DataFrame(row[2].split('|'), columns=['data'])
        df_aux = df_aux[: -1]
        n = len(df_aux)
        df2 = pd.DataFrame(np.nan, index=range(0, n), columns=['x', 'y', 'z'])
        index += n
        # u = pd.date_range(row[0], results[ind + 1][0], periods=n + 1, closed='left')
        # df.set_index(u, inplace=True)
        for fila in df_aux['data']:
            u = fila.split(',')
            df2.loc[cont, :] = u
            cont += 1

        df2['x'] = pd.to_numeric(df2.x)
        df2['y'] = pd.to_numeric(df2.y)
        df2['z'] = pd.to_numeric(df2.z)

        # GRÁFICAS
        # ax = plt.gca()
        # df2.plot(kind='line', y='x', color='blue', ax=ax)
        # df2.plot(kind='line', y='y', color='red', ax=ax)
        # df2.plot(kind='line', y='z', color='green', ax=ax)
        # plt.show()

        # TRANSFORMAR A WAVELETTES
        ñ = ['x', 'y', 'z']
        for u in ñ:
            signal = list(df2[u])
            rec, list_coeff = fx.lowpassfilter(signal, params_freq.thresh, params_freq.wavelet_name)

            # EXTRACCIÓN DE CARACTERÍSTICAS POR BANDA DE FRECUENCIA
            features = []
            artf_var = pd.DataFrame()
            for in_freq, coeff in enumerate(list_coeff):
                features += fx.get_features(coeff)
                # INGENIERÍA DE CARACTERÍSTICAS
                # Originales = ['Entropy', 'Amount zero crossing', 'Amount mean crossing', 'n5', 'n25', 'n75', 'n95', 'median', 'mean', 'std', 'var', 'rms']
                # Artificiales = [ poly2, squares, roots, exp]
                artf_var += fx.artificials_variables(features)

                # GUARDAR EN NUEVA BASE DE DATOS LAS CARACTERÍSTICAS
                db = MySQLdb.connect(host='192.168.3.53', port=3306, user='ariel', passwd='hstech2018', db='SVIA_MCL')
                cursor = db.cursor()

                query = "INSERT INTO features (fecha, valv, eje, freq, var, valor) VALUES (%s,%s,%s,%s,%s,%s);"
                valores = []
                for vr, vl in enumerate(artf_var):
                    valores += (df['fecha'].values, valv, ñ.index(u), in_freq, vr, vl)
                cursor.executemany(query, valores)
                db.commit()

print('')
