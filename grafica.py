import matplotlib.pyplot as plt
import pickle


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


ruta = '/media/arielmardones/HS/SensoVal/'
inercia = ruta + '/inercia.txt'
with open(inercia, 'rb') as fl:
    score = pickle.load(fl)

plot_elbow(score)


