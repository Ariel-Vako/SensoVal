from __future__ import division

"""Consulta Archivos SVIA MQTT"""
__author__ = "Ariel Mardones"
__copyright__ = "Copyright 2019-05-07, highservice"
__version__ = "1.0"

import sys, argparse, csv, os, MySQLdb, time
from datetime import datetime, timedelta
import struct

db = MySQLdb.connect(host='192.168.3.38', port=3306, user='root', passwd='hstech2018!', db='SVIA_MCL')
# ("hstech.sinc.cl", "jsanhueza", "Hstech2018.-)", "ssi_mlp_sag2")
cursor = db.cursor()

query = "id_reg AS ID, time AS Timespam, id_sensor as Sensor, cast( data_sensor AS CHAR) as Data FROM svia_data WHERE id_sensor = 6 ORDER BY id_reg DESC LIMIT 1"

cursor.execute(query)
results = cursor.fetchall()
lista_y = []
for row in results:
    for x in range(540):
        lista_y.append((ord(row[0][x * 2]) << 8) + ord(row[0][x * 2 + 1]) - 2 ** 15)
    # turn = struct.unpack("540h",row[0])
    print(lista_y)
    exit()

cursor = db.cursor()

db = MySQLdb.connect("192.168.3.38", "root", "hstech2018", "ssi_mlp_sag2")
cursor = db.cursor()

cursor.execute("SELECT id_reg, data_sensor , time \
	FROM svia_data \
	WHERE (id_sensor_data IN (1) AND estado_data = 134217727 \
	AND (fecha_reg BETWEEN %s AND %s) ) \
	ORDER BY fecha_reg ASC \
	LIMIT 5000", (startDate, endDate))

results = cursor.fetchall()
feeding_ring = process(results)

