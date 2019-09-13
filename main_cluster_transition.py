import params
import clusters as grp
import pickle
import seaborn as sns
import feature_enginering as fe
import numpy as np

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

# Extracción de características
features = fe.get_sensoval_features(aperturas)
# Generación de variables artificales continuas
df = fe.artificial_variables(features)
# Limpieza de datos
df_clean = fe.clean(df)

# Componentes Principales
caract, pca = grp.componentes_principales(df_clean, params.no_pca)
print(f'Varianza Explicada: {100 * np.round(np.sum(pca.explained_variance_ratio_), 4)}%')

# Métricas para determinar cantidad de cluster y método a utilizar
df_metrics_pca = grp.métricas(caract)
df_metrics_clean = grp.métricas(df_clean)

# Clustering
all_cluster = grp.clustering(caract, no_cluster=params.no_cluster)
print('')
