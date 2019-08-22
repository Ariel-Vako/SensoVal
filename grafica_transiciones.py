import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

sns.set(style="whitegrid")

valvula = 6
ruta = f'/media/arielmardones/HS/SensoVal/Datos/val{valvula}/'
file_apertura = ruta + f'Aperturas_val{valvula}'
file_cierre = ruta + f'Cierres_val{valvula}'

with open(file_apertura, 'rb') as fl:
    aperturas = pickle.load(fl)

with open(file_cierre, 'rb') as fl2:
    cierres = pickle.load(fl2)

# Gráfica conjunta de aperturas
f = plt.figure()
axes = f.add_subplot(121)

sns.despine(f, left=True, bottom=True)

for i in range(len(aperturas)):
    n = len(aperturas[i])
    x = np.linspace(0, n - 1, n)
    axes.scatter(x, aperturas[i], alpha=0.3)

# Gráfica conjunta de aperturas

axes = f.add_subplot(122)

sns.despine(f, left=True, bottom=True)

for i in range(len(cierres)):
    n = len(cierres[i])
    x = np.linspace(0, n - 1, n)
    axes.scatter(x, cierres[i], alpha=0.3)

plt.show()
