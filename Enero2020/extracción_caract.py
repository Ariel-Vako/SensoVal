import numpy as np
import pywt
import scipy
import scipy.optimize
import scipy.stats
import collections


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
    maximo = np.max(list_values)
    minimo = np.min(list_values)
    kurtosis = scipy.stats.kurtosis(list_values, bias=False)
    skewness = scipy.stats.skew(list_values, bias=False)
    return [n5, n25, n75, n95, median, mean, std, maximo, minimo, kurtosis, skewness]


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


def features_from_wavelet(data):
    # 132 columnas = 3 ejes * 4 wavelets * 11 vectores de coeficientes.
    features = []
    for eje in ['x', 'y', 'z']:
        signal = data[eje].values
        for wavelet in ['db1', 'db6', 'db8', 'db10']:
            coeff = pywt.wavedec(signal, wavelet, mode="per", level=10)
            for coeficiente in coeff:
                features += get_features(coeficiente)
    return features
