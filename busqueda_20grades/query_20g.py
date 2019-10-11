from influxdb import InfluxDBClient
import pandas as pd
import pickle
import os.path
import maya
import clustering as clu
import datetime as dt
from datetime import datetime


def mapeo_fechas(cadena):
    return maya.MayaDT.from_rfc3339(cadena).datetime()


def query(valvula):
    cliente = InfluxDBClient(host='192.168.0.178', port=8086, username='', password='', database='SVALVIA_MCL')
    consulta = "SELECT angulo_sensor as Angulo, time as Fecha FROM angulos_svia WHERE angulo_sensor < '21' AND angulo_sensor  > '19' AND id_sensor = '{}'".format(valvula)
    resultado = cliente.query(consulta)
    cliente.close()

    df = pd.DataFrame(list(resultado.get_points()))
    fechas = list(map(mapeo_fechas, df['Fecha']))

    return df['Angulo'], fechas

def diferencia_fecha(fecha1, fecha2):


if __name__ == '__main__':
    valvulas = [6, 8, 9]

    for val in valvulas:
        angulo, fecha = query(val)
