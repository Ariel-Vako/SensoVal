import pickle
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import matplotlib.pyplot as plt


ruta = '/home/arielmardones/Documentos/Respaldo-Ariel/SensoVal/Datos/'
svr_pickle = ruta + 'saveVar.txt'
with open(svr_pickle, 'r') as f:
    X, y, svr_final, reduced_data= pickle.load(f)

sfs = SFS(svr_final,
          k_features=10,
          forward=True,
          floating=False,
          verbose=2,
          scoring='r2',
          cv=0,
          n_jobs=-1)

sfs = sfs.fit(X, y)

print(sfs.k_score_)
print(sfs.k_feature_names_)

fig1 = plot_sfs(sfs.get_metric_dict(), kind='std_dev')

plt.ylim([0.8, 1])
plt.title('Sequential Forward Selection')
plt.grid()
plt.show()



print('')
