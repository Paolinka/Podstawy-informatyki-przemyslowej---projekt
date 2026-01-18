import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import utils

#----------------------------------- CZĘŚĆ (i) -----------------------------------------------------------

# ZADANIE 1


# Wczytanie danych
df = pd.read_excel('L-4/przebieg_norm.xlsx')

# ZADANIE 2
y = df["y (T)"].values.astype(float)
u = df["u (Pg)"].values.astype(float)
z = df["z (Tz)"].values.astype(float)

dt = 1

# wektor wyjścia
Y = y[1:] - y[:-1]

# macierz regresji
X = np.column_stack([
    u[:-1],
    z[:-1] - y[:-1]
])

# estymacja MNK
theta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
theta1, theta2 = theta

# odzyskanie parametrów fizycznych
c_hat = dt / theta1
d_hat = theta2 / theta1

print(f"c_hat = {c_hat:.3f}")
print(f"d_hat = {d_hat:.3f}")


# ZADANIE 3
h_list = [1, 5, 10, 20]
predictions = {}

for h in h_list:
    predictions[h] = utils.predict_with_horizon(
        y=y,
        u=u,
        z=z,
        c=c_hat,
        d=d_hat,
        h=h,
        dt=dt
    )

# ZADANIE 4
# Wspólny wektor czasu
t = np.arange(len(y))

# Wykres zbiorczy predykcji
utils.combo_prediction_plot(t, h_list, y, predictions)
# Wykresy osobne predykcji
utils.separate_prediction_plots(t, h_list, y, predictions)

#----------------------------------- CZĘŚĆ (ii) -----------------------------------------------------------


# ZADANIE 1
differences = utils.count_differences(y, h_list, predictions)
# Wykres zbiorczy różnic
utils.combo_plot_differences(t, h_list, differences)
# Wykresy osobne różnic
utils.separate_plots_differences(t, h_list, differences)

# ZADANIE 2
df = pd.read_excel('L-4/przebieg_zab.xlsx')

# Dane zaburzonego przebiegu
y_zab = df["y (T)"].values.astype(float)
u_zab = df["u (Pg)"].values.astype(float)
z_zab = df["z (Tz)"].values.astype(float)

# Predykcje dla zaburzonego przebiegu
predictions_zab = {}
for h in h_list:
    predictions_zab[h] = utils.predict_with_horizon(
        y=y_zab,
        u=u_zab,
        z=z_zab,
        c=c_hat,
        d=d_hat,
        h=h,
        dt=dt
    )

# różnice dla zaburzonego przebiegu
differences_zab = utils.count_differences(y_zab, h_list, predictions_zab)
t_zab = np.arange(len(y_zab))

# Wykres zbiorczy różnic dla zaburzonego przebiegu
utils.combo_plot_differences(t_zab, h_list, differences_zab)

