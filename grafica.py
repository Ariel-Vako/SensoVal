import matplotlib.pyplot as plt
import pickle
import matplotlib
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator
from datetime import datetime

matplotlib.use('Agg')


# Gráfica de codo mensual para escoger n
def plot_elbow(score):
    escalamiento = [1.0, 1e15, 1e18, 1e16, 1e18, 1e20, 1e20, 1e18, 1e19, 1e19, 1e19]
    for i, valor in enumerate(score):
        plt.subplot(2, 6, i + 1, frameon=False)
        plt.plot([inercias / escalamiento[i] for inercias in valor], color="b", alpha=0.3, linestyle=":", marker='o', markersize=5)
        plt.title(f'Mes {(i + 6) % 12}')
        plt.grid(b=True, which='major', color='#666666')
        plt.grid(b=True, which='minor', color='#999999', alpha=0.2, linestyle='--')
        plt.minorticks_on()
        xmax = min(len(score[i]) - 1, 15)
        plt.xlim([0, xmax])
    plt.show()
    return


def gráfica_transición(fechas, angulos, *args):
    print(f'[{datetime.now()}] Inicio: Gráfica Transición')
    if angulos.size != 0:
        dia = mdates.SecondLocator(interval=2)
        dia_formato = mdates.DateFormatter('%d-%H:%M:%S')

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.xaxis.set_major_locator(dia)
        ax.xaxis.set_major_formatter(dia_formato)
        ax.set_xlim([fechas[0], fechas[-1]])  # - timedelta(days=1)

        ax.grid(b=True, which='major', color='#666666')
        ax.grid(b=True, which='minor', color='#999999', alpha=0.4, linestyle='--')
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.minorticks_on()
        plt.xticks(rotation=45)

        ax.set_xlabel('Tiempo (s)', fontsize=16)
        ax.set_ylabel('Ángulo (°)', fontsize=16)

        ax.scatter(fechas, angulos, alpha=0.3)
        if args:
            ax.set_title(f'Transición {fechas[0]}', fontsize=18)
            fig.savefig(f'/media/arielmardones/HS/SensoVal/Imágenes/val6/{args[0]} + {str(fechas[0].year) + "-" + str(fechas[0].month) + "-" + str(fechas[0].day) + " " + str(fechas[0].hour) + ":" + str(fechas[0].minute)}')
        else:
            ax.set_title(f'Transición {fechas[0]}', fontsize=18)
            fig.savefig(f'/media/arielmardones/HS/SensoVal/Imágenes/val6/{str(fechas[0].year) + "-" + str(fechas[0].month) + "-" + str(fechas[0].day) + " " + str(fechas[0].hour) + ":" + str(fechas[0].minute)}')
    print(f'[{datetime.now()}] Fin: Gráfica Transición')
    return


if __name__ == '__main__':
    ruta = '/media/arielmardones/HS/SensoVal/'
    inercia = ruta + '/inercia.txt'
    with open(inercia, 'rb') as fl:
        score = pickle.load(fl)

    plot_elbow(score)
