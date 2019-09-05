import pickle
import seaborn as sns
import numpy as np
import collections
import scipy.stats
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd


def calculate_entropy(list_values):
    counter_values = collections.Counter(list_values).most_common()
    probabilities = [elem[1] / len(list_values) for elem in counter_values]
    entropy = scipy.stats.entropy(probabilities)
    return entropy


def calculate_statistics(list_values):
    n5 = np.nanpercentile(list_values, 5)
    n25 = np.nanpercentile(list_values, 25)
    n75 = np.nanpercentile(list_values, 75)
    n95 = np.nanpercentile(list_values, 95)
    median = np.nanpercentile(list_values, 50)
    mean = np.nanmean(list_values)
    std = np.nanstd(list_values)
    var = np.nanvar(list_values)
    rms = np.nanmean(np.sqrt(list_values ** 2))
    return [n5, n25, n75, n95, median, mean, std, var, rms]


def calculate_crossings(list_values):
    zero_crossing_indices = np.nonzero(np.diff(np.array(list_values) > 0))[0]
    no_zero_crossings = len(zero_crossing_indices)
    mean_crossing_indices = np.nonzero(np.diff(np.array(list_values) > np.nanmean(list_values)))[0]
    no_mean_crossings = len(mean_crossing_indices)
    return [no_zero_crossings, no_mean_crossings]


def get_features(list_values):
    entropy = calculate_entropy(list_values)
    crossings = calculate_crossings(list_values)
    statistics = calculate_statistics(list_values)
    return [entropy] + crossings + statistics


def artificial_variables(features):
    original_features = pd.DataFrame(features, columns=['Entropy', 'Amount zero crossing', 'Amount mean crossing', 'n5', 'n25', 'n75', 'n95', 'median', 'mean', 'std', 'var', 'rms'])
    plastic = []
    for char in features:
        plastic.append()
    return


def get_sensoval_features(listado_transiente):
    features = []
    for curva in listado_transiente:
        features += get_features(curva)
    return features


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
