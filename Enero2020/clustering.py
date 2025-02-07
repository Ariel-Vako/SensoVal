import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans, MiniBatchKMeans, \
    AffinityPropagation, MeanShift, \
    estimate_bandwidth, spectral_clustering, \
    AgglomerativeClustering, DBSCAN, OPTICS, Birch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import sklearn.metrics
from sklearn.preprocessing import StandardScaler


def clustering_precierre(signal_features, no_cluster=7):
    # kmeans = KMeans(n_clusters=no_cluster, random_state=0).fit(signal_features)
    # mini_kmeans = MiniBatchKMeans(n_clusters=no_cluster, random_state=0, max_iter=10).fit(signal_features)
    # af = AffinityPropagation(preference=-10).fit(signal_features)
    #
    # # --- Mean Shift
    # bandwidth = estimate_bandwidth(signal_features, quantile=0.8)
    # ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    # ms.fit(signal_features)
    # # ms.labels_.max()
    #
    # # --- Spectral Clustering
    # affinity = np.exp(-euclidean_distances(signal_features) / np.std(signal_features))
    # labels_sc = spectral_clustering(affinity, n_clusters=no_cluster, eigen_solver='arpack')
    #
    # # --- Agglomerative Clustering - Solo métrica euclidiana
    # clustering_ward = AgglomerativeClustering(linkage='ward', n_clusters=no_cluster)
    # clustering_ward.fit(signal_features)
    # clustering_average = AgglomerativeClustering(linkage='average', n_clusters=no_cluster)
    # clustering_average.fit(signal_features)
    # clustering_complete = AgglomerativeClustering(linkage='complete', n_clusters=no_cluster)
    # clustering_complete.fit(signal_features)
    # clustering_single = AgglomerativeClustering(linkage='single', n_clusters=no_cluster)
    # clustering_single.fit(signal_features)
    #
    # # --- DBSCAN
    # db = DBSCAN(eps=8, min_samples=6).fit(signal_features)
    #
    # # --- OPTICS
    # optics = OPTICS(min_samples=6, xi=.05, min_cluster_size=.1)
    # optics.fit(signal_features)
    #
    # # --- Birch
    # brc = Birch(branching_factor=100, n_clusters=no_cluster, threshold=20, compute_labels=True)
    # brc.fit(signal_features)

    # --- Agglomerative Clustering - Solo métrica euclidiana
    clustering_complete = AgglomerativeClustering(linkage='single', n_clusters=no_cluster)
    clustering_complete.fit(signal_features)

    return clustering_complete


def componentes_principales(df_features, n):
    # features = preprocessing.normalize(features)
    df_features_standard = StandardScaler().fit_transform(df_features)
    pca = PCA(n_components=n)
    caract = pca.fit_transform(df_features_standard)
    return caract, pca


def graficar_pca(matriz, labels, i):
    método = ['KMeans', 'Mini Batch KMeans', 'Affinity Propagation', 'Mean Shift', 'Spectral Clustering', 'Hierarchical clustering: Ward',
              'Hierarchical clustering: Average', 'Hierarchical clustering: Complete', 'Hierarchical clustering: Single',
              'DBSCAN', 'OPTICS', 'Birch']
    x = matriz[:, 0]
    y = matriz[:, 1]

    groups = {'0': {'pos': (5, 5), 'color': (0.267004, 0.004874, 0.329415)},
              '1': {'pos': (15, 55), 'color': (0.229739, 0.322361, 0.545706)},
              '2': {'pos': (-15, -15), 'color': (0.127568, 0.566949, 0.550556)},
              '3': {'pos': (200, -5), 'color': (0.369214, 0.788888, 0.382914)},
              '4': {'pos': (55, 5), 'color': (0.993248, 0.906157, 0.143936)}}

    # morado      : (0.267004, 0.004874, 0.329415)
    # azul        : (0.229739, 0.322361, 0.545706)
    # verde       : (0.127568, 0.566949, 0.550556)
    # verde pasto : (0.369214, 0.788888, 0.382914)
    # amarillo    : (0.993248, 0.906157, 0.143936)

    # etiquetas_agrupadas = [k for k, it in groupby(sorted(labels))]

    fig, ax = plt.subplots(figsize=(14, 10))
    # plt.gcf().canvas.set_window_title(f'Removing high frequency noise with DWT - Cicle {ciclo}')
    scatter = ax.scatter(x, y, c=labels, alpha=0.3)
    ax.set_title(f'PCA: {método[i]}', fontsize=18)
    ax.set_ylabel('PCA2', fontsize=16)
    ax.set_xlabel('PCA1', fontsize=16)
    # ax.set_xlim(-50, 250)
    # ax.set_ylim(-50, 150)
    ax.grid(b=True, which='major', color='#666666')
    ax.grid(b=True, which='minor', color='#999999', alpha=0.4, linestyle='--')
    ax.minorticks_on()

    for label, value in groups.items():
        plt.annotate(label,
                     value['pos'],
                     horizontalalignment='center',
                     verticalalignment='center',
                     size=15,
                     weight='bold',
                     color=value['color'],
                     backgroundcolor='black')

    legend1 = ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
    ax.add_artist(legend1)

    # plt.show()
    # path = params.ruta + '/gráficas-pca'
    path = '/media/arielmardones/HS/SensoSAG/flex'
    fig.savefig(f'{path}/{método[i]}.png')
    # plt.close('all')
    return


def métricas(signal_features, no_cluster):
    n = no_cluster
    u = range(2, n)
    size = len(u)

    # Método para definición de método de clustering y cantidad de clusters
    df_kmeans = pd.DataFrame(data=np.full([size, 3], np.nan), columns=['ss kmeans', 'ch kmeas', 'db kmeans'])
    df_mbatch = pd.DataFrame(data=np.full([size, 3], np.nan), columns=['ss mbatch', 'ch mbatch', 'db mbatch'])
    # df_affin = pd.DataFrame(data=np.full([size, 3], np.nan), columns=['ss affin', 'ch affin', 'db affin'])
    df_ward = pd.DataFrame(data=np.full([size, 3], np.nan), columns=['ss Ward', 'ch Ward', 'db Ward'])
    df_aver = pd.DataFrame(data=np.full([size, 3], np.nan), columns=['ss Average', 'ch Average', 'db Average'])
    df_comp = pd.DataFrame(data=np.full([size, 3], np.nan), columns=['ss Complete', 'ch Complete', 'db Complete'])
    df_single = pd.DataFrame(data=np.full([size, 3], np.nan), columns=['ss Single', 'ch Single', 'db Single'])
    df_birch = pd.DataFrame(data=np.full([size, 3], np.nan), columns=['ss Birch', 'ch Birch', 'db Birch'])

    for i in u:
        cluster_kmeans = KMeans(n_clusters=i, random_state=0).fit(signal_features)
        df_kmeans.iloc[i - 2, :] = unidimensional_metrics(signal_features, cluster_kmeans.labels_)

        cluster_mnbatch = MiniBatchKMeans(n_clusters=i, random_state=0, max_iter=10).fit(signal_features)
        df_mbatch.iloc[i - 2, :] = unidimensional_metrics(signal_features, cluster_mnbatch.labels_)

        # desv = np.std(signal_features, axis=1)
        # if isinstance(desv, pd.Series):
        #     desv = desv.values
        # affinity = np.exp(-euclidean_distances(signal_features) / desv)
        # labels_sc = spectral_clustering(affinity, n_clusters=i, eigen_solver='amg')
        # df_affin.iloc[i - 2, :] = unidimensional_metrics(signal_features, labels_sc)

        clustering_ward = AgglomerativeClustering(linkage='ward', n_clusters=i)
        cluster_ward = clustering_ward.fit(signal_features)
        df_ward.iloc[i - 2, :] = unidimensional_metrics(signal_features, cluster_ward.labels_)

        clustering_average = AgglomerativeClustering(linkage='average', n_clusters=i)
        cluster_average = clustering_average.fit(signal_features)
        df_aver.iloc[i - 2, :] = unidimensional_metrics(signal_features, cluster_average.labels_)

        clustering_complete = AgglomerativeClustering(linkage='complete', n_clusters=i)
        cluster_complete = clustering_complete.fit(signal_features)
        df_comp.iloc[i - 2, :] = unidimensional_metrics(signal_features, cluster_complete.labels_)

        clustering_single = AgglomerativeClustering(linkage='single', n_clusters=i)
        cluster_single = clustering_single.fit(signal_features)
        df_single.iloc[i - 2, :] = unidimensional_metrics(signal_features, cluster_single.labels_)

        brc = Birch(branching_factor=100, n_clusters=i, threshold=20, compute_labels=True)
        cluster_birch = brc.fit(signal_features)
        df_birch.iloc[i - 2, :] = unidimensional_metrics(signal_features, cluster_birch.labels_)

    df = pd.concat([df_kmeans, df_mbatch, df_ward, df_aver, df_comp, df_single, df_birch], axis=1, sort=False)
    df = df.reindex(sorted(df.columns), axis=1)
    df['N° Cluster'] = list(u)
    df.set_index('N° Cluster', inplace=True)
    return df


def unidimensional_metrics(signal, labels):
    l = pd.DataFrame(labels)
    if l.nunique()[0] > 1:
        # Calinski Harabasz score (Variance Ratio Criterion):  [-inf , inf] == [Worst, Best]
        ch = sklearn.metrics.calinski_harabasz_score(signal, labels)

        # Davies Bouldin score: [0, inf] == [Best, Worst]
        db = sklearn.metrics.davies_bouldin_score(signal, labels)

        # Silhouette Score: [-1, 0 ,1] == [Worst, Overlapping, Best]
        ss = sklearn.metrics.silhouette_score(signal, labels, metric='euclidean')

        return [ss, ch, db]
    else:
        return [np.nan, np.nan, np.nan]

def my_r2(y_true, y_pred):
    numerador = 0
    for i, j in zip(y_true, y_pred):
        numerador += (i-j)**2

    denominador =0
    for z in y_true:
        denominador += (z - np.mean(y_true))**2

    if denominador==0:
        return 0
    else:
        return 1- numerador/denominador