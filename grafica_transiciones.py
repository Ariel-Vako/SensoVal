import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


# import busqueda_paralela
def logistica(t, b, miu, k, a, ñ):
    return k + (a - k) / (1 + np.exp(-b * (t - ñ))) ** (1 / miu)


sns.set(style="whitegrid")

# busqueda_paralela
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

# Formato de aperturas
# Eliminar transiciones incompletas
aperturas0 = []
for i in range(len(aperturas)):
    if all(item < -10 for item in aperturas[i][0:4]):
        aperturas0.append(aperturas[i])

# Alinear al final todas las listas
u = max([len(s) for s in aperturas])

for i in range(len(aperturas0)):
    n = len(aperturas0[i])
    x = np.linspace(0, n - 1, n)
    axes.scatter(x, aperturas0[i], alpha=0.3)
    axes.set_xlabel('Tiempo [s]', fontsize=10)
    axes.set_ylabel('Ángulo [°]', fontsize=10)

# Gráfica conjunta de aperturas

axes = f.add_subplot(122)

sns.despine(f, left=True, bottom=True)

cierres0 = []
for i in range(len(cierres)):
    n = len(cierres[i])
    x = np.linspace(0, n - 1, n)
    if all(item < -10 for item in cierres[i][n - 4:n]):
        cierres0.append(cierres[i])
        axes.scatter(x, cierres[i], alpha=0.3)
        axes.set_xlabel('Tiempo [s]', fontsize=10)
        axes.set_ylabel('Ángulo [°]', fontsize=10)

plt.show()

# Construir array de pares ordenados para
# Aperturas

# Cierres

# Ajuste de Regresiones logísticas
u = np.arange(4500, 10001, 1, dtype=int)
popt2, pcov = curve_fit(logistica, u, df2['y'], bounds=([0.0005, 0.001], [0.001, 2]))
plt.plot(u, logistica(u, *popt2), 'g--')
print(popt2)
print('')
