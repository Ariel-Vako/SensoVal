from influxdb import InfluxDBClient
import pandas as pd

client = InfluxDBClient(host='192.168.0.178', port=8086, username='', password='', database='SVALVIA_MCL')

result = client.query("SELECT mean(angulo_sensor) as Promedio_Angulo, stddev(angulo_sensor) as Desviacion_Estandar   FROM angulos_svia WHERE id_sensor = '6' GROUP BY time(30d) ")
df = pd.DataFrame(list(result.get_points()))

print('')