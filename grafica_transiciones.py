import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
import pandas as pd


# import busqueda_paralela
def logistica_y(param, t, y):
    k = param[0]
    a = param[1]
    b = param[2]
    ñ = param[3]
    miu = param[4]
    # k = 38
    return k + (a - k) / (1 + np.exp(-b * (t - ñ))) ** (1 / miu) - y


def logistica(param, t):
    k = param[0]
    a = param[1]
    b = param[2]
    ñ = param[3]
    miu = param[4]
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

aperturas1 = []
for i in range(len(aperturas0)):
    na = len(aperturas0[i])
    x = np.linspace(0, na - 1, na)
    aperturas1.append(list(zip(x, aperturas[i])))
    axes.scatter(x, aperturas0[i], alpha=0.3)
    axes.set_xlabel('Tiempo [s]', fontsize=10)
    axes.set_ylabel('Ángulo [°]', fontsize=10)

# Construir DataFrames de pares ordenados
# Aperturas (Flat_list)
flat_apert = [item for sublist in aperturas1 for item in sublist]
dfa = pd.DataFrame(flat_apert, columns=['x', 'angulo'])

# Ajuste de Regresiones logísticas para transiciones
x0 = [39.0, -40.0, 1.0, 1.0, 1.0]
for i in range(300):
    # Bootstrapping: Resampling with replacement
    sample_index = np.random.choice(range(len(dfa)), 100)
    par_open = least_squares(logistica_y, x0, args=(dfa['x'].loc[sample_index].values, dfa['angulo'].loc[sample_index].values), loss='soft_l1', f_scale=0.1, bounds=([38, -100.0, -np.inf, 0, -np.inf], [np.inf, -39, 2, np.inf, np.inf]))
    # plt.plot(dfa['x'].values, logistica(dfa['x'].values, *par_open), 'b--')
    xa = np.linspace(0, 29, 30)
    plt.plot(xa, logistica(par_open.x, xa), color='grey', alpha=0.2, zorder=1)

par_open = least_squares(logistica_y, x0, args=(dfa['x'].values, dfa['angulo'].values), loss='soft_l1', f_scale=0.1, bounds=([38, -100.0, -np.inf, 0, -np.inf], [np.inf, -39, 2, np.inf, np.inf]))
plt.plot(xa, logistica(par_open.x, xa), color='red', alpha=0.4, zorder=1)

# Gráfica conjunta de aperturas
axes = f.add_subplot(122)

sns.despine(f, left=True, bottom=True)

cierres0 = []
for i in range(len(cierres)):
    nc = len(cierres[i])
    x = np.linspace(0, nc - 1, nc)
    if all(item < -10 for item in cierres[i][nc - 4:nc]):
        cierres0.append(list(zip(x, cierres[i])))
        axes.scatter(x, cierres[i], alpha=0.3)
        axes.set_xlabel('Tiempo [$s$]', fontsize=10)
        axes.set_ylabel('Ángulo [$°$]', fontsize=10)

# Cierres
flat_cierre = [item for sublist in cierres0 for item in sublist]
dfc = pd.DataFrame(flat_cierre, columns=['x', 'angulo'])

# Ajuste de Regresiones logísticas para transiciones
x0 = [39.0, -40.0, 1.0, 1.0, 1.0]
par_open = least_squares(logistica_y, x0, args=(dfc['x'].values, dfc['angulo'].values), loss='soft_l1', f_scale=0.1, bounds=([38, -100.0, -np.inf, 0, -np.inf], [np.inf, -39, 2, np.inf, np.inf]))
# plt.plot(dfa['x'].values, logistica(dfa['x'].values, *par_open), 'b--')
xc = np.linspace(0, 29, 30)
plt.plot(xc, logistica(par_open.x, xa), '-', label='Robust Ftting')

plt.show()

print('')
