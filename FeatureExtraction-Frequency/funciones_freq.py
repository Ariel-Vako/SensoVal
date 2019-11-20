import numpy as np
import pywt
import MySQLdb
import scipy
import scipy.optimize
import collections
import datetime
import params
from scipy.optimize import least_squares
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def lowpassfilter(signal, thresh=0.63, wavelet="sym7"):
    thresh = thresh * np.nanmax(signal)
    coeff = pywt.wavedec(signal, wavelet, mode="per")  # , level=6)
    coeff[1:] = (pywt.threshold(i, value=thresh, mode="soft") for i in coeff[1:])
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


def consulta_acellz(start_date, end_date, cantidad=5000):
    db = MySQLdb.connect("hstech.sinc.cl", "jsanhueza", "Hstech2018.-)", "ssi_mlp_sag2")
    cursor = db.cursor()
    cursor.execute(
        "SELECT dataZ , fecha_reg FROM Data_Sensor WHERE id_sensor_data IN (3) AND (estado_data = 134217727 OR estado_data = 134217726) AND fecha_reg BETWEEN %s AND %s ORDER BY fecha_reg ASC LIMIT %s",
        (start_date, end_date, cantidad))
    results = cursor.fetchall()
    db.close()
    return results


def extraer_blob(row):
    dates = []
    sample = []
    n = len(row[0]) // 2

    for x in range(n):
        sample.append(float((row[0][x * 2] << 8) + row[0][x * 2 + 1] - 2 ** 15) / 2 ** 8)

    dates.append(row[1])

    return sample, dates


def fundamental(t, amplitud, frecuencia, desfase, desplazamiento_y):
    return amplitud * np.sin(2 * np.pi * frecuencia * t + desfase) - desplazamiento_y


def residuos(t, rec_signal, amplitud, frecuencia, desfase, desplazamiento_y):
    return fundamental(t, amplitud, frecuencia, desfase, desplazamiento_y) - rec_signal


def robust_fitting(signal):
    # Opciones optimización robusta:
    # [linear, huber, soft_l1, cauchy, arctan]
    n = len(signal)
    x0 = [1, 1 / (n * 0.52), 0, -0.5]
    t = np.linspace(0, n, n)
    res_robust = least_squares(error_seno, x0, loss='soft_l1', f_scale=0.1, args=(t, signal), bounds=([0.5, 1 / (0.96 * n), -3.5, -1], [1., 1 / (0.41 * n), 3.5, 1]))
    return res_robust


def toe_average(frecuencia_, raw_impacts_, delta_theta):
    periodo = 1 / frecuencia_
    j = 0
    while j + periodo < len(raw_impacts_):
        raw_impacts_[j] += raw_impacts_[int(periodo) + j]
        j += 1
    impactos = []
    t = 0
    inicio = int(np.ceil((np.pi - delta_theta) / (2 * np.pi * frecuencia_)))
    if inicio < 0:
        inicio = 0
    fin = int((3 * np.pi / 2 - delta_theta) / (2 * np.pi * frecuencia_)) + 1
    if fin > 540:
        fin = 540
    print(inicio, fin)
    angulos = 2 * np.pi * frecuencia_ * np.array(range(inicio, fin)) + delta_theta
    angulos_grad = angulos * 180 / np.pi
    impacto_ponderación = raw_impacts_[inicio: fin]
    toe = np.round((angulos_grad @ impacto_ponderación) / sum(impacto_ponderación), 1)
    return toe, inicio, fin, raw_impacts_


def plot_ajuste(señal_rec, seno2, inicio, fin, raw_impacts_, toe_time, toe, i):
    fig, ax = plt.subplots(figsize=(12, 8))
    rec = señal_rec
    ax.plot(rec, 'k', label='DWT smoothing', linewidth=2)
    ax.plot(raw_impacts_, 'r', label='Distance l1', linewidth=2, alpha=0.5)
    ax.plot(seno2, '--g', label='MCC Robusto', linewidth=1, alpha=0.7)
    ax.legend()
    ax.set_title(f'Ángulo: {np.round(toe, 1)} at {np.round(toe_time, 1)}', fontsize=18)
    ax.set_ylabel('Signal Amplitude', fontsize=16)
    ax.set_xlabel('Time', fontsize=16)
    ax.grid(b=True, which='major', color='#666666')
    ax.grid(b=True, which='minor', color='#999999', alpha=0.4, linestyle='--')
    ax.minorticks_on()
    ax.axvspan(inicio, fin - 1, alpha=0.5, color='#98FB98')
    plt.axvline(x=toe_time, color='NAVY')
    ax.set_xlim([0, 540])
    plt.show()
    fig.savefig(f'/media/arielmardones/HS/SensoSAG/flex/Imágenes/Toe/Ciclo {i}.png')
    plt.close('all')
    return


def error_seno(x0, t, dwt):
    return x0[0] * np.sin(2 * np.pi * x0[1] * t + x0[2]) - x0[3] - dwt
