# -- coding: utf-8 --

import pickle
from Enero2020 import clustering
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, Normalizer, StandardScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold
import pandas as pd
from sklearn.semi_supervised import LabelSpreading
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from os import path
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
import pomegranate as pg
from sklearn.metrics import classification_report, accuracy_score


import warnings

warnings.simplefilter('ignore')


def variance_threshold_selector(data, threshold=0.5):
    selector = VarianceThreshold(threshold)
    selector.fit(data)
    return data[data.columns[selector.get_support(indices=True)]]


valvulas = [6, 8, 9]
ruta_windows = 'C:/Users/ariel/OneDrive/Documentos/HStech/SensoVal/Datos pre-cierre/'
if path.exists(ruta_windows):
    ruta6 = ruta8 = ruta9 = ruta = ruta2 = ruta_windows
else:
    # Lectura características válvula 6
    ruta6 = f'/home/arielmardones/Documentos/Respaldo-Ariel/SensoVal/Datos/val{valvulas[0]}/'
    # Lectura características válvula 8
    ruta8 = f'/home/arielmardones/Documentos/Respaldo-Ariel/SensoVal/Datos/val{valvulas[1]}/'
    # Lectura características válvula 9
    ruta9 = f'/home/arielmardones/Documentos/Respaldo-Ariel/SensoVal/Datos/val{valvulas[2]}/'

    ruta2 = f'/home/arielmardones/Documentos/Respaldo-Ariel/SensoVal/Datos/'

    ruta = f'/home/arielmardones/Documentos/Respaldo-Ariel/SensoVal/Datos/'

file6 = ruta6 + f'Precierre_val{valvulas[0]}'

with open(file6, 'rb') as rf:
    df6 = pickle.load(rf)

file8 = ruta8 + f'Precierre_val{valvulas[1]}'

with open(file8, 'rb') as rf:
    df8 = pickle.load(rf)

file9 = ruta9 + f'Precierre_val{valvulas[2]}'

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

precierre = ruta2 + f'Precierre'
with open(precierre, 'wb') as fl:
    pickle.dump(df, fl)

file = ruta + f'Precierre'

with open(file, 'rb') as rf:
    df = pickle.load(rf)

df.drop([df.index[33], df.index[34], df.index[35], df.index[36], df.index[37], df.index[38]], inplace=True)
class_le = LabelEncoder()
df['valvula'] = class_le.fit_transform(df['valvula'].values)
x = df.loc[:, df.columns[0]:df.columns[-2]]

# Centrado y escalado por desviación estándar
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
# Casos conocidos de fallas. Cambio de valor de algunas etiquetas
# Buenas
reduced_data.loc[0, 'etiqueta'] = 1
reduced_data.loc[1, 'etiqueta'] = 1
reduced_data.loc[2, 'etiqueta'] = 1
reduced_data.loc[3, 'etiqueta'] = 1
reduced_data.loc[4, 'etiqueta'] = 1
reduced_data.loc[5, 'etiqueta'] = 1
reduced_data.loc[6, 'etiqueta'] = 1
reduced_data.loc[7, 'etiqueta'] = 1
reduced_data.loc[8, 'etiqueta'] = 1
reduced_data.loc[9, 'etiqueta'] = 1
reduced_data.loc[10, 'etiqueta'] = 1
reduced_data.loc[26, 'etiqueta'] = 1
reduced_data.loc[62, 'etiqueta'] = 1
reduced_data.loc[63, 'etiqueta'] = 1
reduced_data.loc[64, 'etiqueta'] = 1
reduced_data.loc[95, 'etiqueta'] = 1
reduced_data.loc[96, 'etiqueta'] = 1
reduced_data.loc[97, 'etiqueta'] = 1
reduced_data.loc[98, 'etiqueta'] = 1

# Malas
reduced_data.loc[15, 'etiqueta'] = 0
reduced_data.loc[23, 'etiqueta'] = 0
reduced_data.loc[24, 'etiqueta'] = 0
reduced_data.loc[25, 'etiqueta'] = 0
reduced_data.loc[28, 'etiqueta'] = 0
reduced_data.loc[29, 'etiqueta'] = 0
reduced_data.loc[30, 'etiqueta'] = 0
reduced_data.loc[31, 'etiqueta'] = 0

# TODO: Quitar párrafo después de comprobar el cross validation
xx = reduced_data[reduced_data['etiqueta']!=-1].loc[:, reduced_data.columns[0]:reduced_data.columns[-3]]
yy = reduced_data[reduced_data['etiqueta']!=-1]['etiqueta']

mrl = LogisticRegression()
mrl_scores = cross_val_score(mrl, xx, yy, cv=10).mean()

rgn = svm.SVC()
rgn_scores = cross_val_score(rgn, xx, yy, cv=10).mean()

gnb = GaussianNB()
gnb_scores = cross_val_score(gnb, xx, yy, cv=10).mean()

print()





x_train, x_test = train_test_split(reduced_data, test_size=0.1, random_state=3)
# aux_train = x_train[x_train['valvula'] == 0]
x_train.sort_index(inplace=True)
x_test.sort_index(inplace=True)

# Rename variables
X = x_train[x_train['etiqueta'] != -1].loc[:, x_train.columns[0]:x_train.columns[-3]]
# y = x_train[x_train['etiqueta'] != -1]['Spreading']
y = x_train[x_train['etiqueta'] != -1]['etiqueta']
test = x_test[x_test['etiqueta'] != -1].loc[:, x_train.columns[0]:x_train.columns[-3]]

# Modelos semi-supervisado para las etiquetas
# ls_model = LabelSpreading(kernel='knn', max_iter=80, alpha=0.1, n_neighbors=12)

# ls_model = LabelSpreading()
# ls = ls_model.fit(x_train.loc[:, x_train.columns[0]:x_train.columns[-2]], x_train['etiqueta'])
#
# nb = pg.NaiveBayes.from_samples(pg.BernoulliDistribution, verbose=True)
#
# x_train['Spreading'] = ls.transduction_

# lp_model = LabelPropagation(kernel='knn', max_iter=40)
# lp = lp_model.fit(x_train.loc[:, x_train.columns[0]:x_train.columns[-2]], x_train['etiqueta'])


# Logistic Regression
mrl = LogisticRegressionCV(cv=10, random_state=0, class_weight="auto").fit(X, y)
prediction_lr = mrl.predict_proba(test)
print(mrl.score(X,y))
# Support Vector Regression
svr = svm.SVR(class_weight='balanced', # penalize
              probability=True)
svr.fit(X, y)
prediction_svr = svr.predict(test)



# Gaussian Naive Bayes
gnb = GaussianNB()
prediction_gnb = gnb.fit(X, y).predict(test)

aux2 = pd.DataFrame(prediction_lr, columns=[0, 1])
test['Prob Fallo (LR)'] = aux2[1].values
test['Prob Fallo (SVR)'] = prediction_svr
test['Prob Fallo (GNB)'] = prediction_gnb
test['etiqueta'] = x_test[x_test['etiqueta'] != -1]['etiqueta']
test['fechas'] = fechas[0][x_test.index]
x_train['fechas'] = fechas[0][x_train.index]

print(prediction)
# u2 = pd.DataFrame(fechas.values, columns=['fechas'])
# u2['valvula']= reduced_data['valvula']
# u2['etiqueta']= reduced_data['etiqueta']
# x_train['Spreading']= ls.transduction_
# # x_train['Propagation']= lp.transduction_
