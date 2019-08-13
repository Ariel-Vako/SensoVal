import sys, argparse, csv, os, MySQLdb, time
import pandas as pd
from datetime import datetime, timedelta
import struct

db = MySQLdb.connect(host='192.168.0.178',
                     port=3306,
                     user="amardones",
                     password="AMARDONES.2019!",
                     db="SVIA_MCL")
cursor = db.cursor()

# consulta = "SELECT id_reg AS ID, time AS Timespam, id_sensor as Sensor, cast( data_sensor AS CHAR) as Data FROM svia_data WHERE id_sensor = 6 ORDER BY id_reg ASC LIMIT 10;"
consulta = "SELECT time AS Timespam, cast( data_sensor AS CHAR) FROM svia_data WHERE id_sensor = 6 ORDER BY id_reg ASC LIMIT 10;"
cursor.execute(consulta)
results = cursor.fetchall()

df = pd.DataFrame(columns=('fecha', 'x', 'y', 'z'))
cont = 0
for row in results:
    xyz = row[1].split('|')
    df_aux = pd.DataFrame(xyz)
    for fila in df_aux:
        u = fila.split(',')
        df['x'].iloc[cont] = u[0]
        df['y'].iloc[cont] = u[0]
        df['z'].iloc[cont] = u[0]
        cont += 1
db.close()
