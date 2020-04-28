import pickle

import matplotlib.pyplot as plt
import pandas as pd
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn import svm


def likestepwise(svr, x, y, num_features):
    sfs = SFS(svr,
              k_features=num_features,
              forward=True,
              floating=False,
              verbose=2,
              scoring='r2',
              cv=0,
              n_jobs=-1)

    sfs = sfs.fit(x, y)

    print(f'\nR2: {sfs.k_score_:.3f}')
    print(f'Nombre Variables: {sfs.k_feature_names_}')

    plot_sfs(sfs.get_metric_dict(), kind='std_dev')

    plt.ylim([0.65, 1])
    plt.title('Sequential Forward Selection')
    plt.grid()
    plt.show()

    return sfs


ruta = '/home/arielmardones/Documentos/Respaldo-Ariel/SensoVal/Datos/'
svr_pickle = ruta + 'saveVar.txt'
with open(svr_pickle, 'rb') as f:
    X, y, svr_final, reduced_data, fechas = pickle.load(f)

# sfs1 = likestepwise(svr_final, X, y, 10)
# sfs2 = likestepwise(svr_final, X, y, 3)

col_names = ['n° x Cero-cD2-db1-x', 'n° x Media-cD2-db1-x', 'n° x Cero-cD4-db6-x', 'n° x Media-cD4-db6-x', 'n25-cD5-db8-x', 'n° x Cero-cD6-db8-y', 'n° x Cero-cD4-db8-y', 'n95-cD8-db6-z', 'minimo-cD7-db8-z', 'n° x Media-cD0-db8-z']
# col_names = ['n° x Cero-cD4-db6-x', 'n25-cD5-db8-x', 'n° x Cero-cD4-db8-y']
X2 = X[col_names]
svr_final = svm.SVR(epsilon=0.2)
svr_final.fit(X2, y)
print(f'R2 con {len(col_names)} variables: {svr_final.score(X2, y):.3f}')

svr_final_pickle = ruta + 'mfinal.txt'
with open(svr_final_pickle, 'wb')as ff:
    pickle.dump(svr_final, ff)
# Calculo de la nueva zona Hazard
# scores = []
# hazard_zone = []
# for i in range(100):
#     x_train, x_test = train_test_split(X2, test_size=0.2, random_state=i)
#     x_train.sort_index(inplace=True)
#     x_test.sort_index(inplace=True)
#
#     y_train = y[x_train.index.values]
#     y_test = y[x_test.index.values]
#
#     try:
#         svr_final.fit(x_train, y_train)
#         prediction_svr = svr_final.predict(x_test)
#         fpr_svr, tpr_svr, thresholds_svr = roc_curve(y_test, prediction_svr)
#         optimal_threshold_svr = thresholds_svr[np.argmax(tpr_svr - fpr_svr)]
#         auc_svr = roc_auc_score(y_test, prediction_svr)
#
#         scores.append(auc_svr)
#         hazard_zone.append(optimal_threshold_svr)
#     except ValueError:
#         pass

# hz = np.mean(hazard_zone) # 0.6574675615042034 ; 0.3425324384957966
# print(np.mean(hazard_zone))


# import matplotlib.pyplot as plt
# from sklearn.metrics import roc_curve
# from sklearn.model_selection import train_test_split
#
# x_train, x_test, y_train, y_test = train_test_split(X2, y, test_size=0.2, random_state=35)
# prediction_final2 = svr_final.predict(x_test)
# fpr_svr, tpr_svr, thresholds_svr = roc_curve(y_test, prediction_final2)
# # Gráfica ROC AUC
# # plot the roc curve for the models
# plt.close()
# plt.plot(fpr_svr, tpr_svr, marker='.', label='SVM')
# # axis labels
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# # show the legend
# plt.legend()
# plt.grid()
# # show the plot
# plt.savefig(ruta + 'plot.png')

# Pronostico sobre datos sin etiquetas
xx = reduced_data[reduced_data['etiqueta'] == -1]
yy = reduced_data[reduced_data['etiqueta'] == -1]['etiqueta']
prediction_final = svr_final.predict(xx[col_names])

resumen = pd.DataFrame(prediction_final, columns=['SVR'])
resumen['indice'] = xx.index.values
resumen['fechas'] = fechas[0][xx.index.values].values
resumen.set_index('indice', inplace=True)

print('')
