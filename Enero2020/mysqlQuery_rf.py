from datetime import timedelta

import MySQLdb
import numpy as np
import pandas as pd


def query_mysql(fecha_fin):
    fecha_inicio = fecha_fin - timedelta(minutes=5)

    db = MySQLdb.connect(host='192.168.0.178',
                         port=3306,
                         user="amardones",
                         password="AMARDONES.2019!",
                         db="SVIA_MCL")
    cursor = db.cursor()

    # consulta = "SELECT id_reg AS ID, time AS Timespam, id_sensor as Sensor, cast( data_sensor AS CHAR) as Data FROM svia_data WHERE id_sensor = 6 ORDER BY id_reg ASC LIMIT 10;"
    consulta = "SELECT time AS Timespam, cast( data_sensor AS CHAR) FROM svia_data WHERE id_sensor = 6 AND time  >= '{}' AND time<= '{}';".format(fecha_inicio, fecha_fin)
    cursor.execute(consulta)
    db.close()
    results = cursor.fetchall()

    cont = 0
    index = 0
    df = pd.DataFrame(columns=('x', 'y', 'z'))
    # u = pd.Series()
    for ind, row in enumerate(results):
        df_aux = pd.DataFrame(row[1].split('|'), columns=['data'])
        df_aux = df_aux[: -1]
        n = len(df_aux)
        df = df.append(pd.DataFrame(np.nan, index=range(index, index + n), columns=df.columns))
        index += n
        # u = pd.date_range(row[0], results[ind + 1][0], periods=n + 1, closed='left')
        # df.set_index(u, inplace=True)
        for fila in df_aux['data']:
            u = fila.split(',')
            df.loc[cont, :] = u
            cont += 1

    df['x'] = pd.to_numeric(df.x)
    df['y'] = pd.to_numeric(df.y)
    df['z'] = pd.to_numeric(df.z)

    return df


if __name__ == '__main__':
    df = query_mysql()
    print()
