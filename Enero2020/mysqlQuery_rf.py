from datetime import timedelta, datetime
import MySQLdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import urllib3


def query_mysql(fecha_fin, minutos_antes):
    fecha_inicio = fecha_fin - timedelta(minutes=minutos_antes)
    print(f'{fecha_inicio} & {fecha_fin}')
    http = urllib3.PoolManager()
    if minutos_antes > 0:
        url = "http://innotech.selfip.com:8282/consulta_ssvv.php?ini='{}'&fin='{}'&id=6".format(fecha_inicio, fecha_fin)
    else:
        url = "http://innotech.selfip.com:8282/consulta_ssvv.php?ini='{}'&fin='{}'&id=6".format(fecha_fin, fecha_inicio)
    r = http.request('GET', url)
    resultado = list(str(r.data).split("<br>"))[:-2]

    for i in np.arange(len(resultado), 0, -1):
        if 'r' in resultado[i - 1]:
            del (resultado[i - 1])

    lista = [None] * len(resultado)

    for i, row in enumerate(resultado):
        u = row.split(',')
        lista[i] = u[1:]

    df = pd.DataFrame(lista, columns=['x', 'y', 'z'])
    df['x'] = pd.to_numeric(df.x)
    df['y'] = pd.to_numeric(df.y)
    df['z'] = pd.to_numeric(df.z)

    return df, fecha_inicio


def busqueda_back(path):
    fecha_i = datetime.strptime('2019-01-03T18:10:00', '%Y-%m-%dT%H:%M:%S')

    for i in np.arange(20):
        df, f = query_mysql(fecha_i, 1)
        fecha_i = f
        plt.scatter(df.index.values, df['x'].values, alpha=0.2)
        plt.savefig(f'{path}/Backward-{fecha_i}.png', dpi=500)
        plt.close()
    return


def busqueda_for(path):
    fecha_i = datetime.strptime('2019-01-03T18:10:00', '%Y-%m-%dT%H:%M:%S')

    for i in np.arange(20):
        df, f = query_mysql(fecha_i, -1)
        fecha_i = f
        plt.scatter(df.index.values, df['x'].values, alpha=0.2)
        plt.savefig(f'{path}/Forward-{fecha_i}.png', dpi=500)
        plt.close()

    return


if __name__ == '__main__':
    # Cada 3 minutos
    ruta = '/home/arielmardones/Documentos/Respaldo-Ariel/SensoVal/delay_mysql/val6'
    # Cada un minuto
    # path = '/home/arielmardones/Documentos/Respaldo-Ariel/SensoVal/delay_mysql/val6/minutes_backward'
    busqueda_back(ruta)
    busqueda_for(ruta)
