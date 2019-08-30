from influxdb import InfluxDBClient
import pandas as pd
import matplotlib.pyplot as plt
import mysqlQuery
import numpy as np

client = InfluxDBClient(host='192.168.0.178', port=8086, username='', password='', database='SVALVIA_MCL')

fecha_inicio = '2018-10-22 14:02:00'
fecha_fin = '2018-10-22 14:04:00'
# result = client.query("SELECT mean(angulo_sensor) as Promedio_Angulo, stddev(angulo_sensor) as Desviacion_Estandar   FROM angulos_svia WHERE id_sensor = '6' GROUP BY time(30d) ")
result = client.query("SELECT angulo_sensor as Angulo FROM angulos_svia WHERE id_sensor = '6' AND time < '{}' AND time > '{}'".format(fecha_inicio, fecha_fin))
df1 = pd.DataFrame(list(result.get_points()))

df2 = mysqlQuery.query_mysql(fecha_inicio, fecha_fin)

# Gráficas:
f = plt.figure()
axes = f.add_subplot(121)
# Influx
n1 = len(df1)
x1 = np.linspace(0, n1 - 1, n1)

# Influx
n2 = len(df2)
x2 = np.linspace(0, n2 - 1, n2)

axes.plot(x=x1, y=df1['Angulo'])
axes.plot(x=x2, y=df2['Angulo'])  # TODO: se debe calcular el ángulo
print('')
