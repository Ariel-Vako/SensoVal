from influxdb import InfluxDBClient
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd




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
fecha_inicio  = '2019-01-02 01:19:21'
fecha_fin     = '2019-01-02 01:49:21'


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
# fecha_inicio  = '2019-01-24 09:18:52'
# fecha_fin     = '2019-01-25 09:18:52'

# index: 33 - MALO
# fecha_inicio  = '2019-01-24 22:29:29'
# fecha_fin     = '2019-01-24 22:29:29'




cliente = InfluxDBClient(host='192.168.0.178', port=8086, username='', password='', database='SVALVIA_MCL')
consulta = "SELECT angulo_sensor as angulos FROM angulos_svia WHERE id_sensor = '6' AND time  >= '{}' AND time<= '{}'".format(fecha_inicio, fecha_fin)
resultado = cliente.query(consulta)
cliente.close()

df = pd.DataFrame(list(resultado.get_points()))
df['indice'] = np.arange(len(df))

df.plot(x ='indice', y='angulos', kind = 'scatter', alpha=0.3, s=10)
plt.grid(which='both', axis='both', color='silver', linestyle='--', linewidth=0.5)
plt.show()

# u2 = pd.DataFrame(fechas.values, columns=['fechas'])
# u2['valvula']= reduced_data['valvula']
# u2['etiqueta']= reduced_data['etiqueta']
# x_train['transduction']= ls.transduction_
