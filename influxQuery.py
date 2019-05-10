from influxdb import InfluxDBClient
import pandas as pd

desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 1000)

client = InfluxDBClient(host='192.168.0.178', port=8086, username='', password='', database='SVALVIA_MCL')

# Listado de bases de datos
pd.DataFrame(client.get_list_database())  # [{'name': '_internal'}, {'name': 'SVALVIA_MCL'}, {'name': 'ssi_mlp_sag2_c1'}, {'name': 'Datos_SAG2_MLP'}, {'name': 'FLEETSAFETY'}, {'name': 'SPT_DMH'}]
# Listado de tablas en database
pd.DataFrame(client.get_list_measurements())  # [{'name': 'angulos_svia'}, {'name': 'relay_svia'}]
# Listado de usuarios
pd.DataFrame(client.get_list_users())  # [{'admin': True, 'user': 'crobles'}]

# Listado de nombres de los campos dada una tabla
consulta = "SELECT * FROM /.*/ LIMIT 1"
query_field_names = client.query(consulta)
field_names = list(pd.DataFrame(query_field_names.get_points()).columns.values)


result = client.query("SELECT angulo_sensor as Angulo, id_sensor, status_relay FROM angulos_svia WHERE id_sensor = '6' LIMIT 100")
df = pd.DataFrame(list(result.get_points()))
