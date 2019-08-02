from influxdb import InfluxDBClient
import pandas as pd
import pickle
import os.path
import maya
import clustering as clu
import datetime as dt
from concurrent.futures import ProcessPoolExecutor


def mapeo_fechas(cadena):
    return maya.MayaDT.from_rfc3339(cadena).datetime()


def query_generator_cierre(fecha_fin):
    fecha_inicio = fecha_fin - dt.timedelta(seconds=60)

    cliente = InfluxDBClient(host='192.168.0.178', port=8086, username='', password='', database='SVALVIA_MCL')
    consulta = "SELECT angulo_sensor as Angulo, time as Fecha FROM angulos_svia WHERE id_sensor = '6' AND time  >= '{}' AND time<= '{}'".format(fecha_inicio.strftime("%Y-%m-%d %H:%M:%S"), fecha_fin.strftime("%Y-%m-%d %H:%M:%S"))
    resultado = cliente.query(consulta)
    cliente.close()

    df = pd.DataFrame(list(resultado.get_points()))

    e = ProcessPoolExecutor()
    fechas_cierre = list(e.map(mapeo_fechas, df['Fecha']))

    ii, ifl = ventana(list(df['Angulo']), 0)
    if ii == 0 and ifl == 0:
        return [], []
    else:
        return df['Angulo'][ii - 1:ifl + 1], fechas_cierre[ii - 1: ifl + 1]


def query_generator_apertura(fecha_inicio):
    fecha_fin = fecha_inicio + dt.timedelta(seconds=60)

    cliente = InfluxDBClient(host='192.168.0.178', port=8086, username='', password='', database='SVALVIA_MCL')
    consulta = "SELECT angulo_sensor as Angulo, time as Fecha FROM angulos_svia WHERE id_sensor = '6' AND time  >= '{}' AND time<= '{}'".format(fecha_inicio.strftime("%Y-%m-%d %H:%M:%S"), fecha_fin.strftime("%Y-%m-%d %H:%M:%S"))
    resultado = cliente.query(consulta)
    cliente.close()

    df = pd.DataFrame(list(resultado.get_points()))

    e = ProcessPoolExecutor()
    fechas_apertura = list(e.map(mapeo_fechas, df['Fecha']))

    ii, ifl = ventana(list(df['Angulo']), 1)
    if ii == 0 and ifl == 0:
        return [], []
    else:
        return df['Angulo'][ii:ifl + 1], fechas_apertura[ii: ifl + 1]


def ventana(cadena_angulos, flag):
    if flag == 0:  # cerrando
        index_inicio = len(cadena_angulos)

        i = 0
        buscar_inicio = cadena_angulos[-1]
        while buscar_inicio < 39 and i < len(cadena_angulos):
            index_inicio -= 1
            buscar_inicio = cadena_angulos[index_inicio]
            i += 1

        index_final = 0
        buscar_final = cadena_angulos[0]
        while buscar_final > -40 and index_final < len(cadena_angulos) - 1:
            index_final += 1
            buscar_final = cadena_angulos[index_final]

    else:  # abriendo

        i = 0
        index_inicio = len(cadena_angulos)
        buscar_inicio = cadena_angulos[-1]
        while buscar_inicio > -39 and i < len(cadena_angulos):
            index_inicio -= 1
            buscar_inicio = cadena_angulos[index_inicio]
            i += 1

        index_final = 0
        buscar_final = cadena_angulos[0]
        while buscar_final < 40 and index_final < len(cadena_angulos) - 1:
            index_final += 1
            buscar_final = cadena_angulos[index_final]

    return index_inicio, index_final


if __name__ == '__main__':
    desired_width = 320
    pd.set_option('display.width', desired_width)
    pd.set_option('display.max_columns', 1000)

    # client = InfluxDBClient(host='192.168.0.178', port=8086, username='', password='', database='SVALVIA_MCL')
    # # Listado de bases de datos
    # pd.DataFrame(client.get_list_database())  # [{'name': '_internal'}, {'name': 'SVALVIA_MCL'}, {'name': 'ssi_mlp_sag2_c1'}, {'name': 'Datos_SAG2_MLP'}, {'name': 'FLEETSAFETY'}, {'name': 'SPT_DMH'}]
    # # Listado de tablas en database
    # pd.DataFrame(client.get_list_measurements())  # [{'name': 'angulos_svia'}, {'name': 'relay_svia'}]
    # # Listado de usuarios
    # pd.DataFrame(client.get_list_users())  # [{'admin': True, 'user': 'crobles'}]
    #
    # # Listado de nombres de los campos dada una tabla
    # consulta = "SELECT * FROM angulos_svia LIMIT 1"  # ['angulo_sensor', 'id_sensor', 'time']
    # query_field_names = client.query(consulta)
    # field_names = list(pd.DataFrame(query_field_names.get_points()).columns.values)
    #
    # # Listado de nombres de los campos dada una tabla
    # consulta = "SELECT * FROM relay_svia LIMIT 1"  # ['id_sensor', 'status_relay', 'time']
    # query_field_names = client.query(consulta)
    # field_names = list(pd.DataFrame(query_field_names.get_points()).columns.values)

    # first time
    ruta = '/media/arielmardones/HS/SensoVal/'
    query_bckup = ruta + 'query_influx_sensoVal.txt'
    fecha_file = ruta + 'fechas.txt'
    if not os.path.isfile(query_bckup):
        client = InfluxDBClient(host='192.168.0.178', port=8086, username='', password='', database='SVALVIA_MCL')

        result = client.query("SELECT angulo_sensor as Angulo, time as Fecha FROM angulos_svia WHERE id_sensor = '6' and angulo_sensor<= -40  AND time < '2019-04-30'")
        df = pd.DataFrame(list(result.get_points()))
        # fecha = [maya.MayaDT.from_rfc3339(ee).datetime() for ee in df['Fecha']]

        e = ProcessPoolExecutor()
        fecha = list(e.map(mapeo_fechas, df['Fecha']))
        df['fecha'] = fecha
        df.pop('Fecha')

        with open(query_bckup, 'wb') as fl:
            pickle.dump(df, fl)

        #  Date format: RFC 3339
        with open(fecha_file, 'wb') as fl:
            pickle.dump(fecha, fl)

    with open(query_bckup, 'rb') as rf:
        df = pickle.load(rf)

    with open(fecha_file, 'rb') as rf2:
        fecha = pickle.load(rf2)

    # fig, ax = plt.subplots(figsize=(10, 5))
    # ax.scatter(fecha, df['Angulo'], color='red', alpha=0.3)
    # plt.show()
    #
    # # set title and labels for axes
    # ax.set(xlabel="Fecha",
    #        ylabel="Ángulo de posición",
    #        title="Válvula Abierta")
    #
    # # Malla
    # plt.grid(color='lightgray', linestyle='-', linewidth=0.3)
    #
    # # rotate tick labels
    # plt.setp(ax.get_xticklabels(), rotation=45)

    # Cálculo de número de cluster por mes
    score = clu.n_monthly(fecha)

    # Guardar la inercia de los cluster.- SOLO LA PRIMERA VEZ!
    # inercia = ruta + '/inercia.txt'
    # with open(inercia, 'wb') as fl:
    #     pickle.dump(score, fl)
