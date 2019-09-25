import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
import pandas as pd


def grafica(aperturas, labels_a, cierres, labels_c):
    # Paleta de colores
    custom_palette = ['#131E3A', '#C21807']

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
        axes.scatter(x, aperturas0[i], alpha=0.3, color=custom_palette[labels_a[i]])
        axes.set_xlabel('Tiempo [s]', fontsize=10)
        axes.set_ylabel('Ángulo [°]', fontsize=10)

    # Gráfica conjunta de aperturas
    axes = f.add_subplot(122)

    sns.despine(f, left=True, bottom=True)

    cierres0 = []
    for i in range(len(cierres)):
        nc = len(cierres[i])
        x = np.linspace(0, nc - 1, nc)
        if all(item < -10 for item in cierres[i][nc - 4:nc]):
            cierres0.append(list(zip(x, cierres[i])))
            axes.scatter(x, cierres[i], alpha=0.3, color=custom_palette[labels_a[i]])
            axes.set_xlabel('Tiempo [$s$]', fontsize=10)
            axes.set_ylabel('Ángulo [$°$]', fontsize=10)

    plt.show()
    return
