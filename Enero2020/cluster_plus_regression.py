# -- coding: utf-8 --

import pickle
from Enero2020 import clustering
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, Normalizer, StandardScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold
import pandas as pd
from sklearn.semi_supervised import LabelSpreading
from sklearn.model_selection import train_test_split


def variance_threshold_selector(data, threshold=0.5):
    selector = VarianceThreshold(threshold)
    selector.fit(data)
    return data[data.columns[selector.get_support(indices=True)]]


valvulas = [6, 8, 9]

# Lectura características válvula 6
ruta = f'/home/arielmardones/Documentos/Respaldo-Ariel/SensoVal/Datos/val{valvulas[0]}/'
file6 = ruta + f'Precierre_val{valvulas[0]}'

with open(file6, 'rb') as rf:
    df6 = pickle.load(rf)

# Lectura características válvula 8
ruta = f'/home/arielmardones/Documentos/Respaldo-Ariel/SensoVal/Datos/val{valvulas[1]}/'
file8 = ruta + f'Precierre_val{valvulas[1]}'

with open(file8, 'rb') as rf:
    df8 = pickle.load(rf)

# Lectura características válvula 9
ruta = f'/home/arielmardones/Documentos/Respaldo-Ariel/SensoVal/Datos/val{valvulas[2]}/'
file9 = ruta + f'Precierre_val{valvulas[2]}'

with open(file9, 'rb') as rf:
    df9 = pickle.load(rf)

features_names = []
for eje in ['x', 'y', 'z']:
    for wavelet in ['db1', 'db6', 'db8', 'db10']:
        for coeficiente in ['cA', 'cD9', 'cD8', 'cD7', 'cD6', 'cD5', 'cD4', 'cD3', 'cD2', 'cD1', 'cD0']:
            for features in ['Entropy', 'n° x Cero', 'n° x Media', 'n5', 'n25', 'n75', 'n95', 'median', 'mean', 'std', 'maximo', 'minimo', 'kurtosis', 'skewness']:
                features_names += [features + '-' + coeficiente + '-' + wavelet + '-' + eje]

df = df6.append(df8.append(df9))
aux = features_names + ['valvula']
df.columns = aux

ruta2 = f'/home/arielmardones/Documentos/Respaldo-Ariel/SensoVal/Datos/'
precierre = ruta2 + f'Precierre'
with open(precierre, 'wb') as fl:
    pickle.dump(df, fl)

ruta = f'/home/arielmardones/Documentos/Respaldo-Ariel/SensoVal/Datos/'
file = ruta + f'Precierre'

with open(file, 'rb') as rf:
    df = pickle.load(rf)

class_le = LabelEncoder()
df['valvula'] = class_le.fit_transform(df['valvula'].values)
x = df.loc[:, df.columns[0]:df.columns[-2]]

# Centrado y escalado por desviación estandar
scaler = RobustScaler().fit(x)
# rescaledX = scaler.transform(x)
rescaledX = pd.DataFrame(scaler.transform(x), columns=x.columns)

# scaler = Normalizer().fit(X)
# normalizedX = scaler.transform(X)

# scaler = MinMaxScaler(feature_range=(0, 1))
# rescaledX = scaler.fit_transform(x)

# Remover variables con escada información.
reduced_data = variance_threshold_selector(rescaledX, 0.1)
# rd = reduced_data.sort_index()
fechas = pd.DataFrame(x.index)

# Cluster de 2 y 3 grupos salen escogidos.
# medidas = clustering.métricas(reduced_data, 7)

# Split the data
reduced_data['valvula'] = df['valvula'].values
reduced_data['etiqueta'] = -1
x_train, x_test = train_test_split(reduced_data, test_size=0.1, random_state=42)
# aux_train = x_train[x_train['valvula'] == 0]

# Casos conocidos de fallas. Cambio de valor de algunas etiquetas
x_train.loc['15', 'etiqueta'] = 0

# Modelo semi-supervisado para las etiquetas
lp_model = LabelSpreading(gamma=0.25, max_iter=20)
lp_model.fit(reduced_data.loc[:, reduced_data.columns[0]:reduced_data.columns[-2]], reduced_data['etiqueta'])

predicted_labels = lp_model.transduction_[unlabeled_indices]
true_labels = y[unlabeled_indices]

cluster_02 = clustering.clustering_precierre(df.iloc[:, 0:-2], 2)
cluster_03 = clustering.clustering_precierre(df.iloc[:, 0:-2], 3)

df['etiqueta_2'] = cluster_02.labels_
df['etiqueta_3'] = cluster_03.labels_
print('')
