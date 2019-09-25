import params
import clusters as grp
import pickle
import seaborn as sns
import feature_enginering as fe
import grafica_transicion_por_cluster as grf_cl


def processing(data):
    # Extracción de características
    features = fe.get_sensoval_features(data)
    # Generación de variables artificales continuas
    df = fe.artificial_variables(features)
    # Limpieza de datos
    df_clean = fe.clean(df)
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

df_open = processing(aperturas)
df_close = processing(cierres)
# Componentes Principales
# caract, pca = grp.componentes_principales(df_clean, params.no_pca)
# print(f'Varianza Explicada: {100 * np.round(np.sum(pca.explained_variance_ratio_), 4)}%')

# Métricas para determinar cantidad de cluster y método a utilizar
# df_metrics_pca = grp.métricas(caract)
# df_metrics_clean = grp.métricas(df_clean)

# -------- Conclusiones de la comparación entre PCA y raw data
# Los datos sin PCA obtienen valores de clustering notoriamente mejores a los PCA.
# Esto se entiende dado que se pierde varianza con PCA que proviene de las curvas que representan fallas.

# Los algoritmos de Herarquical Clustering fueron los que entregaron mejor resultados (0.7~).
# Se logra apreciar que las mejores separaciones ocurren con 2 clústers.

# Clustering aperturas
cluster_open = grp.clustering(df_open, no_cluster=params.no_cluster)

# Clustering cierres
cluster_close = grp.clustering(df_close, no_cluster=params.no_cluster)
grf_cl.grafica(df_open, cluster_open.labels_, df_close, cluster_close.labels_)

print('')
