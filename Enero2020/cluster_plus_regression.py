# -- coding: utf-8 --

import pickle

valvulas = [6, 8, 9]

# Lectura características válvula 6
# Trabajo
# ruta = f'/home/arielmardones/Documentos/Respaldo-Ariel/SensoVal/Datos/val{valvulas[0]}/'
# Home
ruta = 'C:/Users/ariel/OneDrive/Documentos/HStech/SensoVal/Datos pre-cierre/'
file6 = ruta + f'Precierre_val{valvulas[0]}'

with open(file6, 'rb') as rf:
    df6 = pickle.load(rf)

# Lectura características válvula 8
# ruta = f'/home/arielmardones/Documentos/Respaldo-Ariel/SensoVal/Datos/val{valvulas[1]}/'
# Home
ruta = 'C:/Users/ariel/OneDrive/Documentos/HStech/SensoVal/Datos pre-cierre/'
file8 = ruta + f'Precierre_val{valvulas[1]}'

with open(file8, 'rb') as rf:
    df8 = pickle.load(rf)

# Lectura características válvula 9
# ruta = f'/home/arielmardones/Documentos/Respaldo-Ariel/SensoVal/Datos/val{valvulas[2]}/'
# Home
ruta = 'C:/Users/ariel/OneDrive/Documentos/HStech/SensoVal/Datos pre-cierre/'
file9 = ruta + f'Precierre_val{valvulas[2]}'

with open(file9, 'rb') as rf:
    df9 = pickle.load(rf)

print('')
