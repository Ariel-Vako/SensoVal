import params
import clusters as grp
import pickle
import seaborn as sns
import feature_enginering as fe
import grafica_transicion_por_cluster as grf_cl
import numpy as np


def processing(data, flag):
    # Extracción de características
    features = fe.get_sensoval_features(data)
    # Generación de variables artificales continuas
    df = fe.artificial_variables(features)
    # Limpieza de datos
    if flag == 0:
        df_clean = fe.clean_close(df)
    elif flag == 1:
        df_clean = fe.clean_open(df)
    return df_clean


sns.set(style="whitegrid")

# busqueda_paralela
valvula = params.valvula
ruta = params.ruta
file_apertura = ruta + f'Aperturas_val{valvula}'
file_cierre = ruta + f'Cierres_val{valvula}'

with open(file_apertura, 'rb') as fl:
    aperturas = pickle.load(fl)

with open(file_cierre, 'rb') as fl2:
    cierres = pickle.load(fl2)

# *** APERTURAS ***
df_clean_open = processing(aperturas, 1)

# Componentes Principales
# caract_open, pca = grp.componentes_principales(df_clean_open, params.no_pca_open)
# print(f'Varianza Explicada: {100 * np.round(np.sum(pca.explained_variance_ratio_), 4)}%')
#
# # Métricas para determinar cantidad de cluster y método a utilizar
# df_metrics_pca = grp.métricas(caract_open, params.no_cluster_cierre)
# df_metrics_clean = grp.métricas(df_clean_open, params.no_cluster_cierre)

# -------- Conclusiones de la comparación entre PCA y raw data para APERTURAS
# Los datos sin PCA obtienen valores de clustering notoriamente mejores a los PCA.
# Esto se entiende dado que se pierde varianza con PCA que proviene de las curvas que representan fallas.

# Los algoritmos de Herarquical Clustering fueron los que entregaron mejor resultados (0.7~).
# Se logra apreciar que las mejores separaciones ocurren con 2 clústers.

# *** CIERRES ***
df_clean_close = processing(cierres, 0)

# Componentes Principales
# caract_close, pca = grp.componentes_principales(df_clean_close, params.no_pca_close)
# print(f'Varianza Explicada: {100 * np.round(np.sum(pca.explained_variance_ratio_), 4)}%')
#
# # Métricas para determinar cantidad de cluster y método a utilizar
# df_metrics_pca_close = grp.métricas(caract_close, params.no_cluster_cierre)
# df_metrics_clean_close = grp.métricas(df_clean_close, params.no_cluster_cierre)

# Clustering aperturas
cluster_open = grp.clustering_open(df_clean_open, no_cluster=params.no_cluster_apertura)

# Clustering cierres
cluster_close = grp.clustering_close(df_clean_close, no_cluster=params.no_cluster_cierre)

# Gráficas
grf_cl.grafica(aperturas, cluster_open.labels_, cierres, cluster_close.labels_)

print('')
