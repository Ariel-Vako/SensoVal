# -- coding: utf-8 --

import pickle
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.feature_selection import VarianceThreshold
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from os import path
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.neural_network import MLPClassifier

import numpy as np

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

    ruta = '/home/arielmardones/Documentos/Respaldo-Ariel/SensoVal/Datos/'

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
# reduced_data.loc[6, 'etiqueta'] = 1
# reduced_data.loc[7, 'etiqueta'] = 1
# reduced_data.loc[8, 'etiqueta'] = 1
# reduced_data.loc[9, 'etiqueta'] = 1
# reduced_data.loc[10, 'etiqueta'] = 1
# reduced_data.loc[26, 'etiqueta'] = 1
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

# #
# xx = reduced_data[reduced_data['etiqueta']!=-1].loc[:, reduced_data.columns[0]:reduced_data.columns[-3]]
# yy = reduced_data[reduced_data['etiqueta']!=-1]['etiqueta']
#
# mrl = LogisticRegression()
# mrl_scores = cross_val_score(mrl, xx, yy, cv=10).mean()
#
# rgn = svm.SVC()
# rgn_scores = cross_val_score(rgn, xx, yy, cv=10).mean()
#
# gnb = GaussianNB()
# gnb_scores = cross_val_score(gnb, xx, yy, cv=10).mean()
#
# print()

# Similar to ross Validation
scores = []
hazard_zone = []
for i in range(50):
    x_train, x_test = train_test_split(reduced_data, test_size=0.25, random_state=i)
    # aux_train = x_train[x_train['valvula'] == 0]
    x_train.sort_index(inplace=True)
    x_test.sort_index(inplace=True)

    # Rename variables
    X = x_train[x_train['etiqueta'] != -1].iloc[:, 0:-3]
    # y = x_train[x_train['etiqueta'] != -1]['Spreading']
    y = x_train[x_train['etiqueta'] != -1]['etiqueta']
    test = x_test[x_test['etiqueta'] != -1].iloc[:, 0:-3]

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
    y_test = x_test[x_test['etiqueta'] != -1]['etiqueta']

    try:
        # NN
        nn = MLPClassifier(solver='lbfgs', alpha=0.1, random_state=30,
                           hidden_layer_sizes=[100, 100])
        nn.fit(X, y)
        prediction_nn = pd.DataFrame(nn.predict_proba(test))
        prediction_nn = prediction_nn[1].values
        fpr_nn, tpr_nn, thresholds_nn = roc_curve(y_test, prediction_nn)
        optimal_threshold_nn = thresholds_nn[np.argmax(tpr_nn - fpr_nn)]
        auc_nn = roc_auc_score(y_test, prediction_nn)
        # print(f'ROC AUC NN: {auc_nn:.1f}')
        # score_nn = nn.score(test, y_test)

        # Logistic Regression
        mrl = LogisticRegressionCV(cv=10, random_state=0, class_weight="auto").fit(X, y)
        prediction_lr = mrl.predict_proba(test)
        aux = pd.DataFrame(prediction_lr, columns=[0, 1])
        prediction_lr = aux[1].values
        fpr_mrl, tpr_mrl, thresholds_mrl = roc_curve(y_test, prediction_lr)
        optimal_threshold_mrl = thresholds_mrl[np.argmax(tpr_mrl - fpr_mrl)]
        auc_mrl = roc_auc_score(y_test, prediction_lr)
        # print(f'ROC AUC Logistic Regression: {auc_mrl:.1f}')
        # score_lr = mrl.score(test, y_test)

        # Support Vector Regression
        svr = svm.SVR(epsilon=0.2)
        svr.fit(X, y)
        prediction_svr = svr.predict(test)
        fpr_svr, tpr_svr, thresholds_svr = roc_curve(y_test, prediction_svr)
        optimal_threshold_svr = thresholds_svr[np.argmax(tpr_svr - fpr_svr)]
        auc_svr = roc_auc_score(y_test, prediction_svr)
        # print(f'ROC AUC Support Vector Reggression: {auc_svr:.1f}')
        # score_svr=svr.score(test, x_test[x_test['etiqueta'] != -1]['etiqueta'])

        # Gaussian Naive Bayes
        gnb = GaussianNB()
        prediction_gnb = gnb.fit(X, y).predict_proba(test)
        aux2 = pd.DataFrame(prediction_gnb, columns=[0, 1])
        prediction_gnb = aux2[1].values
        fpr_gnb, tpr_gnb, thresholds_gnb = roc_curve(y_test, prediction_gnb)
        optimal_threshold_gnb = thresholds_gnb[np.argmax(tpr_gnb - fpr_gnb)]
        auc_gnb = roc_auc_score(y_test, prediction_gnb)
        # print(f'ROC AUC Gaussian Naive Bayes: {auc_gnb:.1f}')
        # score_gnb=gnb.score(test, y_test)

        scores.append([auc_nn, auc_mrl, auc_svr, auc_gnb])
        hazard_zone.append([optimal_threshold_nn, optimal_threshold_mrl, optimal_threshold_svr, optimal_threshold_gnb])
    except ValueError:
        pass
df_scores = pd.DataFrame(scores, columns=['auc_nn', 'auc_mrl', 'auc_svr', 'auc_gnb'])
df_scores.describe()

df_hazard = pd.DataFrame(hazard_zone, columns=['nn', 'mrl', 'svr', 'gnb'])
df_hazard.describe()

# df_tpr = pd.DataFrame(true_positive, columns=['tpr_nn','tpr_mrl','tpr_svr','tpr_gnb'])
# df_tpr.describe()

# Fitting again SVR
X = reduced_data[reduced_data['etiqueta'] != -1].iloc[:, 0:-3]
y = reduced_data[reduced_data['etiqueta'] != -1]['etiqueta']

svr_final = svm.SVR(epsilon=0.2)
svr_final.fit(X, y)
print(f'R2 : {svr_final.score(X, y)}')

svr_pickle = ruta + 'saveVar.txt'
with open(svr_pickle, 'wb') as f:
    pickle.dump([X, y, svr_final, reduced_data, fechas], f)

# prediction_svr_final = svr.predict(test)
# fpr_svr, tpr_svr, thresholds_svr = roc_curve(y_test, prediction_svr)
# auc_svr = roc_auc_score(y_test, prediction_svr)

# test['Prob Fallo (LR)'] = prediction_lr
# test['Prob Fallo (SVR)'] = prediction_svr
# test['Prob Fallo (GNB)'] = prediction_gnb
# test['Prob Fallo (NN)'] = prediction_nn
# test['etiqueta'] = x_test[x_test['etiqueta'] != -1]['etiqueta']
# test['fechas'] = fechas[0][x_test.index]
# x_train['fechas'] = fechas[0][x_train.index]
# zu = test.iloc[:, -6:-1]
#
# # Gráfica ROC AUC
# # plot the roc curve for the models
# plt.plot(fpr_nn, tpr_nn, linestyle='--', label='NN')
# plt.plot(fpr_mrl, tpr_mrl, marker='.', label='Logistic')
# plt.plot(fpr_svr, tpr_svr, marker='.', label='SVM')
# plt.plot(fpr_gnb, tpr_gnb, marker='.', label='Naive Bayes')
# # axis labels
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# # show the legend
# plt.legend()
# # show the plot
# plt.show()
#
# print('')
# u2 = pd.DataFrame(fechas.values, columns=['fechas'])
# u2['valvula']= reduced_data['valvula']
# u2['etiqueta']= reduced_data['etiqueta']
# x_train['Spreading']= ls.transduction_
# # x_train['Propagation']= lp.transduction_
