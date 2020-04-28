from datetime import timedelta, datetime
import pandas as pd
import urllib3
import numpy as np
import matplotlib.pyplot as plt


def query_mysql(fecha_fin, minutos_antes, válvula):
    fecha_inicio = fecha_fin - timedelta(minutes=minutos_antes)

    http = urllib3.PoolManager()
    if minutos_antes > 0:
        url = "http://innotech.selfip.com:8282/consulta_ssvv.php?ini='{}'&fin='{}'&id={}".format(fecha_inicio, fecha_fin, válvula)
    else:
        url = "http://innotech.selfip.com:8282/consulta_ssvv.php?ini='{}'&fin='{}'&id={}".format(fecha_fin, fecha_inicio, válvula)
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

    return df


def grafica(df_plot):
    plt.close('all')
    ax = plt.gca()

    df_plot.plot(kind='line', y='x', ax=ax)
    df_plot.plot(kind='line', y='y', color='red', ax=ax)
    df_plot.plot(kind='line', y='z', color='green', ax=ax)

    plt.grid(True, color='grey', alpha=0.3)
    plt.show()


# index: 14 - BUENO
# fecha_inicio = '2018-11-10 23:22:44'
# fecha_fin    = '2018-11-10 23:28:44'

# index: 15 - MALO
# fecha_inicio = '2018-11-13 13:36:39'
# fecha_fin    = '2018-11-13 13:56:39'

# index: 16 - BUENO
# fecha_inicio = '2018-11-15 14:38:30'
# fecha_fin    = '2018-11-15 14:48:30'

# index: 17 - BUENO
# fecha_inicio = '2018-11-16 06:36:12'
# fecha_fin    = '2018-11-18 08:56:12'

# index: 18 - BUENO
# fecha_inicio = '2018-11-19 14:49:44'
# fecha_fin    = '2018-11-20 06:49:44'

# index: 19 - BUENO
# fecha_inicio = '2018-11-21 09:00:43'
# fecha_fin    = '2018-11-21 9:55:43'

# index: 20 - BUENO
# fecha_inicio = '2018-11-23 21:56:25'
# fecha_fin    = '2018-11-24 21:56:25'

# index: 21 - BUENO
# fecha_inicio = '2018-11-25 11:39:52'
# fecha_fin    = '2018-11-26 00:39:52'

# index: 22 -
# fecha_inicio = '2018-11-26 01:42:12'
# fecha_fin    = '2018-12-01 01:42:12'
# fecha_inicio = '2018-12-01 01:42:12'
# fecha_fin    = '2018-12-06 01:42:12'
# fecha_inicio = '2018-12-06 01:42:12'
# fecha_fin    = '2018-12-11 01:42:12'
# fecha_inicio = '2018-12-11 01:42:12'
# fecha_fin    = '2018-12-13 17:11:12'

# index: 23 - MALO
# fecha_inicio  = '2018-12-13 17:11:17'
# fecha_fin     = '2018-12-15 17:11:17'
# fecha_inicio  = '2018-12-15 17:11:17'
# fecha_fin     = '2018-12-19 17:11:17'
# fecha_inicio  = '2018-12-15 17:11:17'
# fecha_fin     = '2018-12-19 17:11:17'
# fecha_inicio   = '2018-12-24 17:11:17'
# fecha_fin      = '2018-12-27 17:11:17'
# fecha_inicio   = '2018-12-27 11:11:17'
# fecha_fin      = '2018-12-27 13:11:17'

# index: 24 - MALO
# fecha_inicio  = '2018-12-31 03:46:56'
# fecha_fin     = '2019-01-01 11:46:56'
# fecha_inicio  = '2018-12-31 18:12:56'
# fecha_fin     = '2018-12-31 18:15:56'

# index: 25 - MALO
# fecha_inicio  = '2018-12-31 03:52:59'
# fecha_fin     = '2019-01-01 12:52:59'

# index: 26 -
# fecha_inicio  = '2019-01-02 01:19:21'
# fecha_fin     = '2019-01-02 01:49:21'


# index: 28 - MALO
# fecha_inicio  = '2019-01-02 03:15:57'
# fecha_fin     = '2019-01-02 03:22:57'

# index: 29 - MALO
# fecha_inicio  = '2019-01-02 23:22:42'
# fecha_fin     = '2019-01-03 01:30:42'
# fecha_inicio  = '2019-01-03 01:28:42'
# fecha_fin     = '2019-01-03 01:30:42'

# index: 30 - MALO
# fecha_inicio  = '2019-01-03 03:23:44'
# fecha_fin     = '2019-01-03 22:33:44'
# fecha_inicio  = '2019-01-03 21:10:44'
# fecha_fin     = '2019-01-03 21:12:44'

# index: 32 - MALO
# fecha_inicio  = '2019-01-24 08:55:10'
# fecha_fin     = '2019-01-24 22:53:08'

# index: 33 - MALO
# fecha_inicio  = '2019-01-24 09:19:33'
# fecha_fin     = '2019-01-24 22:35:03'

# index: 51
# fecha_inicio  = '2019-04-08 04:51:36'
# fecha_fin     = '2019-04-08 05:57:31'

# index: 55
# fecha_inicio  = '2019-04-28 13:47:13'
# fecha_fin     = '2019-04-29 05:46:44'

# # index: 68
# fecha_inicio  = '2018-10-31 22:52:56'
# fecha_fin     = '2018-11-01 08:34:44'

# index: 71
# fecha_inicio  = '2018-12-17 11:50:17'
# fecha_fin     = '2018-12-17 11:51:38'

# # index: 75
# fecha_inicio  = '2019-01-02 02:00:56'
# fecha_fin     = '2019-01-02 05:21:45'

# index: 90
# fecha_inicio  = '2019-04-20 20:37:28'
# fecha_fin     = '2019-04-21 20:12:35'


# index: 115
# fecha_inicio  = '2019-01-08 12:54:34'
# fecha_fin     = '2019-01-08 15:35:40'


# --- TEST --- #
# index: 4 - BUENO
# fecha_inicio  = '2018-09-20 12:59:30'
# fecha_fin     =

# index: 19 - Feas vibraciones (válvula cerrada)
# fecha_inicio = '2018-11-21 19:00:43'
# fecha_fin = '2018-11-22 19:40:43'


# index: 31 - MALO
# fecha_inicio  = '2019-01-17 07:40:58'
# fecha_fin     =

# index: 36
# fecha_inicio  = '2019-02-26 21:50:26'
# fecha_fin     =

# index: 44
# fecha_inicio  = '2019-03-14 15:32:55'
# fecha_fin     =

# index: 53
# fecha_inicio  = '2019-04-19 17:43:36'
# fecha_fin     =

# index: 67
# fecha_inicio  = '2018-10-31 08:08:56'
# fecha_fin     =

# index: 73
# fecha_inicio  = '2018-12-19 13:12:26'
# fecha_fin     =

# index: 77
# fecha_inicio  = '2019-01-11 21:36:22'
# fecha_fin     =

# index: 94
# fecha_inicio  = '2019-05-26 04:48:26'
# fecha_fin     =

# index: 104
# fecha_inicio  = '2018-10-18 04:16:16'
# fecha_fin     =

# index: 116
# fecha_inicio  = '2019-02-18 05:50:37'
# fecha_fin     =

# index: 117
# fecha_inicio  = '2019-03-05 21:50:51'
# fecha_fin     =


válvula = 6
# fecha_inicio  = '2018-08-31 06:03:50'
# fecha_fin     = '2018-08-31 07:12:09'

# Revisión en las aceleraciones
fecha_fin = datetime.strptime(fecha_fin, '%Y-%m-%d %H:%M:%S')
fecha_inicio = datetime.strptime(fecha_inicio, '%Y-%m-%d %H:%M:%S')
minutos_antes = fecha_fin - fecha_inicio
print(minutos_antes.total_seconds()/60)
df = query_mysql(fecha_fin, minutos_antes.total_seconds() / 60, válvula)
print(df.shape)
grafica(df)
print(fecha_inicio)
# cliente = InfluxDBClient(host='192.168.0.178', port=8086, username='', password='', database='SVALVIA_MCL')
# consulta = "SELECT angulo_sensor as angulos FROM angulos_svia WHERE id_sensor = '6' AND time  >= '{}' AND time<= '{}'".format(fecha_inicio, fecha_fin)
# resultado = cliente.query(consulta)
# cliente.close()
#
# df = pd.DataFrame(list(resultado.get_points()))
# df['indice'] = np.arange(len(df))
#
# df.plot(x ='indice', y='angulos', kind = 'scatter', alpha=0.3, s=10)
# plt.grid(which='both', axis='both', color='silver', linestyle='--', linewidth=0.5)
# plt.show()
