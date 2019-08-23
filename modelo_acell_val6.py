import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
import numpy as np
from scipy.interpolate import CubicSpline


def seno(t, f):
    return 250 * np.sin(2 * np.pi * f * (t - 4500)) + 750


def logistica(t, b, miu):
    return 600 + (-700 - 600) / (1 + np.exp(-b * (t - 4500))) ** (1 / miu)


def cubica(t, a, b, c, d):
    return a * t ** 3 + b * t ** 2 + c * t + d


ruta = '/media/arielmardones/HS/SensoVal/'
file = ruta + 'acel6.txt'

with open(file, 'rb') as fl:
    df = pickle.load(fl)

df['x'] = pd.to_numeric(df.x)
df['y'] = pd.to_numeric(df.y)
df['z'] = pd.to_numeric(df.z)

df.loc[4500:10000].plot()
df2 = df.loc[4500:10000]
plt.grid(b=True, which='major', color='#666666')
# Modelo X
u = np.arange(4500, 10001, 1, dtype=int)
popt, pcov = curve_fit(seno, u, df2['x'], bounds=([8e-5], [0.0001]))
plt.plot(u, seno(u, *popt), 'r--')
print(f'Modelo x: {popt}')

# Modelo Y
model_y = LinearRegression()
u = np.arange(4500, 10001, 1, dtype=int).reshape(-1, 1)
model_y.fit(u, df2['y'])
plt.plot(u, model_y.predict(u), 'r--')
print(f'Modelo y: {model_y.coef_}')
# Modelo_Y2
# u = np.arange(4500, 10001, 1, dtype=int)
# popt2, pcov = curve_fit(logistica, u, df2['y'], bounds=([0.0005, 0.001], [0.001, 2]))
# plt.plot(u, logistica(u, *popt2), 'g--')
# print(popt2)

# Modelo_Y3
# cs = CubicSpline(u[0:-1:50], df2['y'].iloc[0:-1:50])
# plt.plot(u, cs(u), 'k--')

# Modelo_Y4
# u = np.arange(4500, 10001, 1, dtype=int)
# popt4, pcov = curve_fit(cubica, u, df2['y'])
# plt.plot(u, cubica(u, *popt4), 'g--')
# print(popt4)

# Modelo Z
model_z = LinearRegression()
u = np.arange(4500, 10001, 1, dtype=int).reshape(-1, 1)
model_z.fit(u, df2['z'])
plt.plot(u, model_z.predict(u), 'y--')
print(f'Modelo z: {model_z.coef_}')

plt.xlabel('[s]')
plt.ylabel('[mg]')

plt.show()
print('')
