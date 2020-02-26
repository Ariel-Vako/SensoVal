from influxdb import InfluxDBClient
import pandas as pd
import numpy as np
import maya
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import datetime


def mapeo_fechas(cadena):
    return maya.MayaDT.from_rfc3339(cadena).datetime()


def query(valvula):
    fecha_inicio = '2018-07-01'
    cliente = InfluxDBClient(host='192.168.0.178', port=8086, username='', password='', database='SVALVIA_MCL')
    consulta = "SELECT angulo_sensor as Angulo, time as Fecha FROM angulos_svia WHERE angulo_sensor < 0 AND angulo_sensor  > -30 AND id_sensor = '{}' AND time > '{}'".format(valvula, fecha_inicio)
    resultado = cliente.query(consulta)
    cliente.close()

    df = pd.DataFrame(list(resultado.get_points()))
    fechas = list(map(mapeo_fechas, df['Fecha']))
    df['fecha'] = fechas
    df.pop('Fecha')
    return df


def grafica_20g(fechas, angulos):
    dia = mdates.SecondLocator(interval=2)
    dia_formato = mdates.DateFormatter('%d-%H:%M:%S')

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.xaxis.set_major_locator(dia)
    ax.xaxis.set_major_formatter(dia_formato)
    ax.set_xlim([fechas[0], fechas[-1]])

    ax.grid(b=True, which='major', color='#666666')
    ax.grid(b=True, which='minor', color='#999999', alpha=0.4, linestyle='--')
    ax.xaxis.set_minor_locator(MultipleLocator(2))
    ax.minorticks_on()
    plt.xticks(rotation=45)

    ax.set_xlabel('Tiempo (s)', fontsize=16)
    ax.set_ylabel('Ángulo (°)', fontsize=16)

    ax.scatter(fechas, angulos, alpha=0.3)
    ax.set_title(f'Transición {fechas[0]}', fontsize=18)


if __name__ == '__main__':
    valvulas = [6, 8, 9]

    df = query(valvulas[0])

    # Generar nuevo dataframe que contenga muestra de fechas cada "n" tiempo
    segmentación = pd.DataFrame(columns=['inicio', 'fin', 'diferencia'])
    segmentación['inicio'] = df['fecha'][0:-1]
    segmentación['fin'] = segmentación['inicio'].shift(periods=-1)
    segmentación.reset_index(inplace=True)
    segmentación.drop(segmentación.index[-1], inplace=True)

    # Diferencia de fechas en segundos
    segmentación['diferencia'] = segmentación['fin'] - segmentación['inicio']
    segmentación['diferencia'] = segmentación['diferencia'] / np.timedelta64(1, 's')

    # Segmentar DF
    index_inicio = []
    index_fin = []
    cont = 0
    while cont < len(segmentación):
        if segmentación['diferencia'].iloc[cont] < 3:
            index_inicio.append(cont)
            cont += 1
        else:
            cont += 1
            continue

        aux = 0
        flag = False
        while cont < len(segmentación) and flag is False:
            if segmentación['diferencia'].iloc[cont] < 5:
                aux = cont
            else:
                flag = True
            cont += 1

        if aux == 0:
            index_fin.append(cont - 1)
        else:
            index_fin.append(aux + 1)

    # Por guardar listado de eventos en la banda [0,-23]
    # para análisis posteriores
    indexs = pd.DataFrame(list(zip(index_inicio, index_fin, np.array(index_fin) - np.array(index_inicio) + 1)), columns=['Index inicio', 'Index fin', 'Cantidad de eventos'])

    # 0 - 5
    eventos_0to5 = indexs[indexs['Cantidad de eventos'] <= 5]
    eventos_0to5.reset_index(inplace=True)
    eventos_0to5.pop('index')
    # 6 - 25
    eventos_6to25 = indexs[(indexs['Cantidad de eventos'] > 5) & (indexs['Cantidad de eventos'] <= 25)]
    eventos_6to25.reset_index(inplace=True)
    eventos_6to25.pop('index')
    # Over 25
    eventos_over25 = indexs[indexs['Cantidad de eventos'] > 25]
    eventos_over25.reset_index(inplace=True)
    eventos_over25.pop('index')

    # Gráficas 0 - 5 Eventos
    u5 = 2
    indice_inicio5 = eventos_0to5['Index inicio'].iloc[u5]
    indice_fin5 = eventos_0to5['Index fin'].iloc[u5] + 1
    grafica_20g(df['fecha'].iloc[indice_inicio5:indice_fin5].values, df['Angulo'].iloc[indice_inicio5:indice_fin5].values)

    # Gráficas 6 - 25 Eventos
    u25 = 12
    indice_inicio25 = eventos_6to25['Index inicio'].iloc[u25]
    indice_fin25 = eventos_6to25['Index fin'].iloc[u25] + 1
    grafica_20g(df['fecha'].iloc[indice_inicio25:indice_fin25].values, df['Angulo'].iloc[indice_inicio25:indice_fin25].values)

    # Gráficas +26 Eventos
    u26plus = 15
    indice_inicio = eventos_over25['Index inicio'].iloc[u26plus]
    indice_fin = eventos_over25['Index fin'].iloc[u26plus] + 1
    grafica_20g(df['fecha'].iloc[indice_inicio:indice_fin].values, df['Angulo'].iloc[indice_inicio:indice_fin].values)
    print('')
