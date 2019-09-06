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
    col_names = ['Entropy', 'Amount zero crossing', 'Amount mean crossing', 'n5', 'n25', 'n75', 'n95', 'median', 'mean', 'std', 'var', 'rms']
    n = len(col_names)
    original_features = pd.DataFrame(features, columns=col_names)
    poly2 = pd.DataFrame(data=np.full([n, n * (n - 1) // 2], np.nan))
    ratios_upper = pd.DataFrame(data=np.full([n, n * (n - 1) // 2], np.nan))
    ratios_lower = pd.DataFrame(data=np.full([n, n * (n - 1) // 2], np.nan))

    # poly2
    cont = 0
    for i in range(n):
        for j in range(i + 1, n):
            operation = ' * '
            name = col_names[i] + operation + col_names[j]
            poly2.rename(columns={cont: name}, inplace=True)
            poly2[name] = original_features[col_names[i]] * original_features[col_names[j]]
            cont += 1

    # ratio upper
    cont = 0
    for i in range(n):
        for j in range(i + 1, n):
            operation = ' / '
            name = col_names[i] + operation + col_names[j]
            ratios_upper.rename(columns={cont: name}, inplace=True)
            ratios_upper[name] = original_features[col_names[i]] / original_features[col_names[j]]
            cont += 1

    # ratio lower
    cont = 0
    for i in range(n):
        for j in range(i + 1, n):
            operation = ' / '
            name = col_names[j] + operation + col_names[i]
            ratios_lower.rename(columns={cont: name}, inplace=True)
            ratios_lower[name] = original_features[col_names[j]] / original_features[col_names[i]]
            cont += 1

    # Squares
    col_sq = [cn + '^2' for cn in col_names]
    squares = original_features.apply(lambda x: x * x)
    squares.rename(columns=dict(zip(col_names, col_sq)), inplace=True)

    # Squares roots
    col_rt = ['sqrt(' + cn + ')' for cn in col_names]
    roots = original_features.apply(lambda x: np.sqrt(x))
    roots.rename(columns=dict(zip(col_names, col_rt)), inplace=True)

    # Natural logarithm
    col_ln = ['ln(' + cn + ')' for cn in col_names]
    ln = original_features.apply(lambda x: np.log(x))
    ln.rename(columns=dict(zip(col_names, col_ln)), inplace=True)

    # Exponential
    col_exp = ['exp(' + cn + ')' for cn in col_names]
    exp = original_features.apply(lambda x: np.exp(x))
    exp.rename(columns=dict(zip(col_names, col_exp)), inplace=True)

    # Inverse
    col_inv = [cn + 'E-1' for cn in col_names]
    inv = original_features.apply(lambda x: 1 / x)
    inv.rename(columns=dict(zip(col_names, col_inv)), inplace=True)

    df = pd.concat([original_features, poly2, ratios_upper, ratios_lower, squares, roots, exp, inv], axis=1, sort=False)
    return df


def get_sensoval_features(listado_transiente):
    features = []
    for curva in listado_transiente:
        features += [get_features(curva)]
        # features += get_features(curva)
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

features = get_sensoval_features(aperturas)
df = artificial_variables(features)
df_clean = clean(df)
