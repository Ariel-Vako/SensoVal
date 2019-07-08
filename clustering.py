from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def find_clusters(data, no_cluster=7):
    kmeans = KMeans(n_clusters=no_cluster, random_state=0).fit(data)
    return kmeans


def find_n_clusters(data):
    resultado = []
    for i in range(2, len(data) - 1, 1):
        kmeans = find_clusters(data, i)
        resultado.append(kmeans.score())
    return resultado


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
    for mes in range(7, 13, 1):
        data_mes_nreal = []
        for index, fecha in enumerate(data_fechas[inicio, -1]):
            if fecha.month == mes:
                fecha_nreal = from_time_to_natural(fecha)
                data_mes_nreal.append(fecha_nreal)
            else:
                inicio = index + 1
                break
        res.append(find_n_clusters(data_mes_nreal))
    return res


def plotea(score):
    for i, valor in enumerate(score):
        plt.subplot(240 + i)
        plt.plot(valor, color="b", alpha=0.5)
        plt.title(f'Mes {i + 6}')
    plt.show()
    return
