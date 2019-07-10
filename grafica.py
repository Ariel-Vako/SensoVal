import matplotlib.pyplot as plt
import pickle


# Gráfica de codo mensual para escoger n
def plot_elbow(score):
    for i, valor in enumerate(score):
        plt.subplot(240 + i)
        plt.plot(valor, color="b", alpha=0.5)
        plt.title(f'Mes {i + 6}')
    plt.show()
    return


ruta = '/media/arielmardones/HS/SensoVal/'
inercia = ruta + '/inercia.txt'
with open(inercia, 'rb') as fl:
    score = pickle.load(fl)

plot_elbow(score)
