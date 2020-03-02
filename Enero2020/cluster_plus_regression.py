# -- coding: utf-8 --

import pickle
from Enero2020 import clustering

# # Preformateo previo. Se prepara el dataframe para alimentar a los algoritmos de clustering

# # El pre-formateo se ejecuta solamente la primera vez. Luego guardamos un único dataframe con los resultados.
# valvulas = [6, 8, 9]
#
# # Lectura características válvula 6
# ruta = f'/home/arielmardones/Documentos/Respaldo-Ariel/SensoVal/Datos/val{valvulas[0]}/'
# file6 = ruta + f'Precierre_val{valvulas[0]}'
#
# with open(file6, 'rb') as rf:
#     df6 = pickle.load(rf)
#
# # Lectura características válvula 8
# ruta = f'/home/arielmardones/Documentos/Respaldo-Ariel/SensoVal/Datos/val{valvulas[1]}/'
# file8 = ruta + f'Precierre_val{valvulas[1]}'
#
# with open(file8, 'rb') as rf:
#     df8 = pickle.load(rf)
#
# # Lectura características válvula 9
# ruta = f'/home/arielmardones/Documentos/Respaldo-Ariel/SensoVal/Datos/val{valvulas[2]}/'
# file9 = ruta + f'Precierre_val{valvulas[2]}'
#
# with open(file9, 'rb') as rf:
#     df9 = pickle.load(rf)
#
# features_names = []
# for eje in ['x', 'y', 'z']:
#     for wavelet in ['db1', 'db6', 'db8', 'db10']:
#         for coeficiente in ['cA', 'cD9', 'cD8', 'cD7', 'cD6', 'cD5', 'cD4', 'cD3', 'cD2', 'cD1', 'cD0']:
#             for features in ['Entropy', 'n° x Cero', 'n° x Media', 'n5', 'n25', 'n75', 'n95', 'median', 'mean', 'std', 'maximo', 'minimo', 'kurtosis', 'skewness']:
#                 features_names += [features + '-' + coeficiente + '-' + wavelet + '-' + eje]
#
# df = df6.append(df8.append(df9))
# aux = features_names + ['valvula']
# df.columns = aux
#
# ruta2 = f'/home/arielmardones/Documentos/Respaldo-Ariel/SensoVal/Datos/'
# precierre = ruta2 + f'Precierre'
# with open(precierre, 'wb') as fl:
#     pickle.dump(df, fl)

ruta = f'/home/arielmardones/Documentos/Respaldo-Ariel/SensoVal/Datos/'
file = ruta + f'Precierre'

with open(file, 'rb') as rf:
    df = pickle.load(rf)

medidas = clustering.métricas(df.iloc[:, 0:-2], 7)

cluster_02 = clustering.clustering_precierre(df.iloc[:, 0:-2], 2)
cluster_03 = clustering.clustering_precierre(df.iloc[:, 0:-2], 3)
print('')
