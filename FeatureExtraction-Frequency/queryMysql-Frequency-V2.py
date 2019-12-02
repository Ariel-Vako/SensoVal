from __future__ import division
import funciones_freq as fx
import params_freq
import MySQLdb
import pandas as pd
import numpy as np
from multiprocessing import Process
import sys


def query_x_val(valv, first_index):
    id_reg = 0
    last_index = int(first_index) + 3000000  # {6: '14834701', 8: '15317765', 9: '14834717'}
    flag = False
    while not flag:
        db = MySQLdb.connect(host='192.168.3.53', port=3306, user='ariel', passwd='hstech2018', db='SVIA_MCL')
        cursor = db.cursor()
        if id_reg == 0:
            query = "SELECT id_reg, time, cast( data_sensor AS CHAR) FROM svia_data WHERE id_sensor = '{}' AND time > '2018-07-15' AND id_reg>='{}' LIMIT 1;".format(valv, first_index)
        else:
            if not id_reg == last_index:
                query = "SELECT id_reg, time, cast( data_sensor AS CHAR) FROM svia_data WHERE id_sensor = '{}' AND id_reg > '{}' LIMIT 1;".format(valv, id_reg)
            else:
                db.close()
                break

        cursor.execute(query)
        db.close()
        results = cursor.fetchall()
        df = pd.DataFrame(list(results), columns=('id', 'fecha', 'blob'))
        id_reg = df['id'].values[0]
        print(f'Válvula: {valv}, Id: {id_reg}')

        # SEPARAR BLOB POR COORDENADAS
        df_aux = pd.DataFrame(results[0][2].split('|'), columns=['data'])
        df_aux = df_aux[: -1]
        n = len(df_aux)
        df2 = pd.DataFrame(np.nan, index=range(0, n), columns=['x', 'y', 'z'])

        cont = 0
        for fila in df_aux['data']:
            u = fila.split(',')
            df2.loc[cont, :] = u
            cont += 1

        # COORDENADAS A NUMÉRICO
        df2['x'] = pd.to_numeric(df2.x)
        df2['y'] = pd.to_numeric(df2.y)
        df2['z'] = pd.to_numeric(df2.z)

        # TRANSFORMAR A WAVELETTES
        ñ = ['x', 'y', 'z']
        for u in ñ:
            signal = list(df2[u])
            rec, list_coeff = fx.lowpassfilter(signal, params_freq.thresh, params_freq.wavelet_name)

            # EXTRACCIÓN DE CARACTERÍSTICAS POR BANDA DE FRECUENCIA
            features = []
            artf_var = pd.DataFrame()
            for in_freq, coeff in enumerate(list_coeff):
                features = fx.get_features(coeff)
                # INGENIERÍA DE CARACTERÍSTICAS
                # Originales = ['Entropy', 'Amount zero crossing', 'Amount mean crossing', 'n5', 'n25', 'n75', 'n95', 'median', 'mean', 'std', 'var', 'rms']
                # Artificiales = [ poly2, squares, exp]
                artf_var = fx.artificials_variables(features)

                # GUARDAR EN NUEVA BASE DE DATOS LAS CARACTERÍSTICAS
                db = MySQLdb.connect(host='192.168.3.53', port=3306, user='ariel', passwd='hstech2018', db='SVIA_MCL')
                cursor = db.cursor()

                query = "INSERT INTO features (fecha, valv, eje, banda_freq, var, valor) VALUES (%s,%s,%s,%s,%s,%s);"
                valores = []
                fecha = pd.to_datetime(str(df['fecha'].values[0]))
                dt = fecha.strftime('%Y-%m-%d %H:%M:%S')
                for vr, vl in enumerate(artf_var):
                    valores.append((dt, valv, ñ.index(u), in_freq, vr, artf_var[vl].values[0]))
                cursor.executemany(query, valores)
                db.commit()


if __name__ == '__main__':

    processes = []
    for valvula in [6, 8, 9]:
        processes.append(Process(target=query_x_val, args=(valvula, sys.argv[1])))

    for process in processes:
        process.start()

    for process in processes:
        process.join()
