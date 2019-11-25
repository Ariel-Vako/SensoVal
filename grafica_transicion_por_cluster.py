import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def grafica(aperturas, labels_a, cierres, labels_c):
    # Paleta de colores
    custom_palette = ['#131E3A', '#C21807', '#FCD12A', '#29AB87']

    # Gráfica conjunta de aperturas
    f = plt.figure()
    axes = f.add_subplot(121)

    sns.despine(f, left=True, bottom=True)

    # Formato de aperturas
    # Eliminar transiciones incompletas
    # aperturas0 = []
    # for valv in range(len(aperturas)):
    #     if all(item < -10 for item in aperturas[valv][0:4]):
    #         aperturas0.append(aperturas[valv])

    for i in range(len(aperturas)):
        na = len(aperturas[i])
        x = np.linspace(0, na - 1, na)
        axes.scatter(x, aperturas[i], alpha=0.3, color=custom_palette[labels_a[i]])

    axes.set_xlabel('Tiempo [s]', fontsize=10)
    axes.set_ylabel('Ángulo [°]', fontsize=10)
    axes.set_title('Aperturas')

    custom_circle_close = [Line2D([0], [0], marker='o', color=custom_palette[i]) for i in (0, 1)]
    axes.legend(custom_circle_close, ['Bien', 'Falla'], loc="upper right", title="Clases", bbox_to_anchor=(-0.1, 1))

    # Gráfica conjunta de aperturas
    axes = f.add_subplot(122)

    sns.despine(f, left=True, bottom=True)

    # cierres0 = []

    labels_c = np.insert(labels_c, 21, 3)
    labels_c = np.insert(labels_c, 50, 3)
    for i in range(len(cierres)):
        nc = len(cierres[i])
        x = np.linspace(0, nc - 1, nc)
        scatter = axes.scatter(x, cierres[i], alpha=0.3, color=custom_palette[labels_c[i]], label=custom_palette[labels_c[i]])

    axes.set_xlabel('Tiempo [$s$]', fontsize=10)
    axes.set_ylabel('Ángulo [$°$]', fontsize=10)
    axes.set_title('Cierres')

    custom_circle_close = [Line2D([0], [0], marker='o', color=custom_palette[i]) for i in range(len(custom_palette))]
    axes.legend(custom_circle_close, ['Falla', 'Bien', 'Falso positivo', 'Indeterminado'], loc="upper left", title="Clases", bbox_to_anchor=(1, 1))

    plt.show()
    return
