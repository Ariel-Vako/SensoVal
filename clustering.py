from sklearn.cluster import KMeans


def find_clusters(data, no_cluster=7):
    kmeans = KMeans(n_clusters=no_cluster, random_state=0).fit(data)
    return kmeans


def find_n_clusters(data):
    resultado = []
    for i in range(200):
        kmeans = find_clusters(data, i)
        resultado.append(kmeans.score())
    return resultado
