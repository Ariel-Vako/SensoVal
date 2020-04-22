import pickle
import pandas as pd
from datetime import timedelta, datetime
import urllib3
import numpy as np
import pywt
import sys


def query_mysql(fecha_fin, válvula):
    fecha_inicio = fecha_fin - timedelta(minutes=1)

    http = urllib3.PoolManager()
    url = "http://innotech.selfip.com:8282/consulta_ssvv.php?ini='{}'&fin='{}'&id={}".format(fecha_inicio, fecha_fin, válvula)
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


def calculate_statistics(list_values):
    n25 = np.nanpercentile(list_values, 25)
    n95 = np.nanpercentile(list_values, 95)
    minimo = np.min(list_values)
    return [n25, n95, minimo]


def calculate_crossings(list_values):
    zero_crossing_indices = np.nonzero(np.diff(np.array(list_values) > 0))[0]
    no_zero_crossings = len(zero_crossing_indices)
    mean_crossing_indices = np.nonzero(np.diff(np.array(list_values) > np.nanmean(list_values)))[0]
    no_mean_crossings = len(mean_crossing_indices)
    return [no_zero_crossings, no_mean_crossings]


def get_features(list_values):
    crossings = calculate_crossings(list_values)
    statistics = calculate_statistics(list_values)
    return crossings + statistics


def features_from_wavelet(data):
    features = []
    for eje in ['x', 'y', 'z']:
        signal = data[eje].values
        for wavelet in ['db1', 'db6', 'db8']:
            coeff = pywt.wavedec(signal, wavelet, mode="per", level=9)
            for coeficiente in coeff:
                features += get_features(coeficiente)
    return features


def generador_variables(caract):
    features_names = []
    for eje in ['x', 'y', 'z']:
        for wavelet in ['db1', 'db6', 'db8']:
            for coeficiente in ['cA', 'cD8', 'cD7', 'cD6', 'cD5', 'cD4', 'cD3', 'cD2', 'cD1', 'cD0']:
                for features in ['n° x Cero', 'n° x Media', 'n25', 'n95', 'minimo']:
                    features_names += [features + '-' + coeficiente + '-' + wavelet + '-' + eje]

    cct = pd.DataFrame([caract], columns=features_names)

    col_names = ['n° x Cero-cD2-db1-x', 'n° x Media-cD2-db1-x', 'n° x Cero-cD4-db6-x', 'n° x Media-cD4-db6-x', 'n25-cD5-db8-x', 'n° x Cero-cD6-db8-y', 'n° x Cero-cD4-db8-y', 'n95-cD8-db6-z', 'minimo-cD7-db8-z', 'n° x Media-cD0-db8-z']

    x_sinEscalar = cct[col_names]
    centro = [1444.0, 1447.0, 344.5, 347.0, -23.1823254601958, 84.0, 337.0, 18.025913331175992, -32.260307004165206, 4189.5]
    escala = [220.0, 221.75, 50.0, 50.0, 6.052034108582546, 10.5, 55.75, 9.843396280797322, 18.763488215595537, 1498.75]

    x = (x_sinEscalar - centro) / escala

    return x


if __name__ == '__main__':
    ruta = '/home/arielmardones/Documentos/Respaldo-Ariel/SensoVal/Datos/'
    svr_pickle = ruta + 'mfinal.txt'
    with open(svr_pickle, 'rb') as f:
        svr_final = pickle.load(f)

    fecha_utc = datetime.strptime(sys.argv[1], '%Y-%m-%d %H:%M:%S')
    valvula = int(sys.argv[2])

    señal = query_mysql(fecha_utc, valvula)
    features = features_from_wavelet(señal)
    x = generador_variables(features)

    prob = svr_final.predict(x)
    print(f'Probabilidad de fallo: {100-prob[0]*100:.1f}')
