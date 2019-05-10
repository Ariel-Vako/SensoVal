from __future__ import division
"""Consulta Archivos SVIA MQTT"""
__author__      = "Ariel Mardones"
__copyright__   = "Copyright 2019-05-07, highservice"
__version__     = "1.0"

import sys, argparse, csv, os, MySQLdb, time
from datetime import datetime,timedelta
import struct

db = MySQLdb.connect(host='192.168.0.178', port=3306, user='amardones', passwd='AMARDONES.2019!', db='SVIA_MCL')
cursor = db.cursor()

query = "id_reg AS ID, time AS Timespam, id_sensor as Sensor, cast( data_sensor AS CHAR) as Data...
        FROM svia_data
        WHERE id_sensor = 6
        ORDER BY id_reg DESC
        LIMIT 100"

cursor.execute(query2)
results = cursor.fetchall()
lista_y=[]
for row in results:
    for x in range(540):
        lista_y.append((ord(row[0][x*2])<<8)+ord(row[0][x*2+1])-2**15)
    #turn = struct.unpack("540h",row[0])
    print lista_y
    exit()