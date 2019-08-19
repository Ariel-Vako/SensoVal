import sys, argparse, csv, os, MySQLdb, time
import pandas as pd
from datetime import datetime, timedelta
import struct

now = datetime.now()
db = MySQLdb.connect(host='192.168.0.178',
                     port=3306,
                     user="amardones",
                     password="AMARDONES.2019!",
                     db="SVIA_MCL")
cursor = db.cursor()

# consulta = "SELECT id_reg AS ID, time AS Timespam, id_sensor as Sensor, cast( data_sensor AS CHAR) as Data FROM svia_data WHERE id_sensor = 6 ORDER BY id_reg ASC LIMIT 10;"
consulta = "SELECT time AS Timespam, cast( data_sensor AS CHAR) FROM svia_data WHERE id_sensor = 6 ORDER BY id_reg ASC LIMIT 10;"
cursor.execute(consulta)
db.close()
results = cursor.fetchall()


time2 = datetime.now()
print(f'Tiempo respuesta query: {time2 - now}')

df = pd.DataFrame(columns=('x', 'y', 'z'))
for row in results:
    df_aux = pd.DataFrame(row[1].split('|'), columns=['data'])
    df_aux = df_aux[:, -1]
    n = len(df_aux)
    for fila in df_aux['data']:
        u = fila.split(',')
        df = df.append(pd.DataFrame([u], columns=df.columns), ignore_index=True)

time3 = datetime.now()
print(f'Tiempo de transformación a Dataframe: {time3-time2}')
