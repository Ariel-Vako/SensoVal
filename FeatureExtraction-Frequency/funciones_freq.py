import numpy as np
import pywt
import MySQLdb
import scipy
import scipy.optimize
import scipy.stats
import collections
import datetime
import params
from scipy.optimize import least_squares
import matplotlib
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def lowpassfilter(signal, thresh=0.63, wavelet="sym7"):
    thresh = thresh * np.nanmax(signal)
    coeff = pywt.wavedec(signal, wavelet, mode="per", level=2)
    # coeff[1:] = (pywt.threshold(i, value=thresh, mode="soft") for i in coeff[1:])
    reconstructed_signal = pywt.waverec(coeff, wavelet, mode="per")
    return reconstructed_signal, coeff


def grafica(signal, ciclo, reconstructed_signal, dates):
    fecha = datetime.datetime.strftime(dates[0], '%d-%m-%Y ~ %H:%M:%S')
    plt.close('all')
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.gcf().canvas.set_window_title(f'Removing high frequency noise with DWT - Cicle {ciclo}')
    ax.plot(signal, color="b", alpha=0.5, label='original signal')
    rec = reconstructed_signal
    ax.plot(rec, 'k', label='DWT smoothing', linewidth=2)
    # ax.plot(coseno, 'r', label='Sine fit', linewidth=1, linestyle='--')
    # JS
    # ax.plot(js, 'g', label='Sine JS', linewidth=1, linestyle='--')
    ax.legend()
    ax.set_title(f'Cicle {ciclo + 1}: {fecha}', fontsize=18)
    ax.set_ylabel('Signal Amplitude', fontsize=16)
    ax.set_xlabel('Time', fontsize=16)
    ax.grid(b=True, which='major', color='#666666')
    ax.grid(b=True, which='minor', color='#999999', alpha=0.4, linestyle='--')
    ax.minorticks_on()
    # plt.show()
    fig.savefig(f'/media/arielmardones/HS/SensoSAG/flex/Imágenes/Dwt/Ciclo {ciclo}.png')
    return


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


def get_sensosag_features(ecg_data, ecg_labels, waveletname):
    list_features = []
    list_unique_labels = list(set(ecg_labels))
    list_labels = [list_unique_labels.index(elem) for elem in ecg_labels]
    for signal in ecg_data:
        list_coeff = pywt.wavedec(signal, waveletname)
        features = []
        for coeff in list_coeff:
            features += get_features(coeff)
        list_features.append(features)
    return list_features, list_labels


def artificials_variables(features):
    col_names = ['Entropy', 'Amount zero crossing', 'Amount mean crossing', 'n5', 'n25', 'n75', 'n95', 'median', 'mean', 'std', 'var', 'rms']
    n = len(col_names)
    m = 1
    original_features = pd.DataFrame([features], columns=col_names)
    poly2 = pd.DataFrame(data=np.full([m, n * (n - 1) // 2], np.nan))

    # poly2
    cont = 0
    for i in range(n):
        for j in range(i + 1, n):
            operation = ' * '
            name = col_names[i] + operation + col_names[j]
            poly2.rename(columns={cont: name}, inplace=True)
            poly2[name] = original_features[col_names[i]] * original_features[col_names[j]]
            cont += 1

    # Squares
    col_sq = [cn + '^2' for cn in col_names]
    squares = original_features.apply(lambda x: x * x)
    squares.rename(columns=dict(zip(col_names, col_sq)), inplace=True)

    # Exponential
    col_exp = ['exp(' + cn + ')' for cn in col_names]
    exp = original_features.apply(lambda x: np.exp(x / 10e3))
    exp.rename(columns=dict(zip(col_names, col_exp)), inplace=True)

    df = pd.concat([original_features, poly2, squares, exp], axis=1, sort=False)
    return df
