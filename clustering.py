from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import numpy as np
from collections import Counter
# from concurrent.futures import ThreadPoolExecutor

def find_clusters(data, no_cluster=10):
    score_data = np.asarray(data).reshape(-1, 1)

    # kmeans = KMeans(n_clusters=no_cluster, random_state=0).fit(np.array(data).reshape(-1, 1))
    mn_kmeans = MiniBatchKMeans(n_clusters=no_cluster, random_state=0).fit(np.array(data).reshape(-1, 1))

    ss = silhouette_score(score_data, mn_kmeans.labels_, metric='euclidean')
    ch = calinski_harabasz_score(score_data, mn_kmeans.labels_)
    db = davies_bouldin_score(score_data, mn_kmeans.labels_)
    return mn_kmeans, np.array([ss, ch, db])


def find_n_clusters(data):
    scores = np.empty([1,3])
    top = min(20, len(data))

    # scores = list(map(find_clusters, data, *))
    # e = ThreadPoolExecutor()
    # futures = {}
    for i in range(2, top, 1):
        score_cluster = find_clusters(data, i)
        scores  = np.concatenate((scores, [score_cluster]))

    np.delete(scores,0,0)
    return scores


def from_time_to_natural(fecha):
    micro = fecha.microsecond / 10e5
    segundo = fecha.second
    minuto = fecha.minute * 60
    hora = fecha.hour * 86400
    dia = fecha.day * 2.628e+6
    natural = dia + hora + minuto + segundo + micro
    return natural


def n_monthly(data):
    data_fechas = data
    res = []
    inicio = 0
    # 2018
    for mes in range(6, 13, 1):
        data_mes_nreal = []
        for index, fecha in enumerate(data_fechas[inicio: -1]):
            if fecha.month == mes:
                fecha_nreal = from_time_to_natural(fecha)
                data_mes_nreal.append(fecha_nreal)
            else:
                inicio += index
                break
        res.append(find_n_clusters(data_mes_nreal))
    # 20019
    if data_fechas[inicio].year == 2019:
        for mes in range(1, 5, 1):
            data_mes_nreal = []
            for index, fecha in enumerate(data_fechas[inicio: -1]):
                if fecha.month == mes:
                    fecha_nreal = from_time_to_natural(fecha)
                    data_mes_nreal.append(fecha_nreal)
                else:
                    inicio += index
                    break
            res.append(find_n_clusters(data_mes_nreal))
    return res


def cluster(data, n):
    if n != 0:
        kmeans, score = find_clusters(data, n)
    else:
        kmeans = []
    return kmeans


def if_cluster(grupo):
    dicc_indices = {}
    if grupo:
        n = max(grupo.labels_)
        for etiq in range(n + 1):
            if Counter(grupo.labels_ == etiq)[1] > 3:
                vector_valor = np.where(grupo.labels_ == etiq)[0]
                dicc_indices[str(etiq)] = vector_valor

    return dicc_indices


def cluster_monthly(data, dicc):
    data_fechas = data
    diccionario_cluster = {}
    inicio = 0
    # 2018
    for mes in range(6, 13, 1):
        data_mes_nreal = []
        for index, fecha in enumerate(data_fechas[inicio: -1]):
            if fecha.month == mes:
                fecha_nreal = from_time_to_natural(fecha)
                data_mes_nreal.append(fecha_nreal)
            else:
                inicio += index
                n = dicc[f'{data_fechas[inicio - 1].year}' + f'{mes}']
                break

        grupo = cluster(data_mes_nreal, n)
        diccionario_cluster[f'{data_fechas[inicio - 1].year}' + f'{mes}'] = if_cluster(grupo)
    # 20019
    if data_fechas[inicio].year == 2019:
        for mes in range(1, 5, 1):
            data_mes_nreal = []
            for index, fecha in enumerate(data_fechas[inicio: -1]):
                if fecha.month == mes:
                    fecha_nreal = from_time_to_natural(fecha)
                    data_mes_nreal.append(fecha_nreal)
                else:
                    inicio += index
                    n = dicc[f'{data_fechas[inicio - 1].year}' + f'{fecha.month}']
                    break

            grupo = cluster(data_mes_nreal, n)
            diccionario_cluster[f'{data_fechas[inicio - 1].year}' + f'{mes}'] = if_cluster(grupo)
    return diccionario_cluster
