from __future__ import division

"""Consulta Archivos SVIA MQTT"""
__author__ = "Ariel Mardones"
__copyright__ = "Copyright 2019-05-07, highservice"
__version__ = "1.0"

import sys, argparse, csv, os, MySQLdb, time
from datetime import datetime, timedelta
import struct
import pandas as pd
import numpy as np

db = MySQLdb.connect(host='192.168.3.38', port=3306, user='amardones', passwd='hstech2018', db='SVIA_MCL')
cursor = db.cursor()

query = "SELECT id_reg, time, cast( data_sensor AS CHAR) FROM svia_data WHERE id_sensor = 6 LIMIT 10"

cursor.execute(query)
db.close()
results = cursor.fetchall()
df = pd.DataFrame(list(results), columns=('id', 'fecha', 'blob'))

cont = 0
index = 0
for ind, row in enumerate(results):
    df_aux = pd.DataFrame(row[2].split('|'), columns=['data'])
    df_aux = df_aux[: -1]
    n = len(df_aux)
    df = df.append(pd.DataFrame(np.nan, index=range(index, index + n), columns=df.columns))
    index += n
    # u = pd.date_range(row[0], results[ind + 1][0], periods=n + 1, closed='left')
    # df.set_index(u, inplace=True)
    for fila in df_aux['data']:
        u = fila.split(',')
        df.loc[cont, :] = u
        cont += 1

df['x'] = pd.to_numeric(df.x)
df['y'] = pd.to_numeric(df.y)
df['z'] = pd.to_numeric(df.z)

print('')