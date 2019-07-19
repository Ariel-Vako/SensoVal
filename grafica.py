import matplotlib.pyplot as plt
import pickle
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator


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
        xmax = min(len(score[i]) - 1, 20)
        plt.xlim([0, xmax])
    plt.show()
    return


def gráfica_transición(angulos, fechas):
    dia = mdates.DayLocator(interval=2)
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

    ax.set_ylabel('Angle (°)', fontsize=16)
    ax.set_title(f'Transición', fontsize=18)

    scatter = ax.scatter(fechas, angulos, alpha=0.3)

    ax.show()
    return


if __name__ == '__main__':
    ruta = '/media/arielmardones/HS/SensoVal/'
    inercia = ruta + '/inercia.txt'
    with open(inercia, 'rb') as fl:
        score = pickle.load(fl)

    plot_elbow(score)
