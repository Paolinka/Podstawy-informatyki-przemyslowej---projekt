import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

#----------------------------------- CZĘŚĆ (i) -----------------------------------------------------------

# ZADANIE 1
def simulate_model(y0, u, z, c, d, dt=1.0):
    """
    Symulacja modelu:
    c*y_dot + d*y = u + d*z
    zdyskretyzowanego metodą Eulera
    """
    N = len(u)
    y_hat = np.full(N, np.nan)
    y_hat[0] = y0

    for n in range(N - 1):
        y_hat[n + 1] = y_hat[n] + (dt / c) * (u[n] + d * z[n] - d * y_hat[n])

    return y_hat


def predict_with_horizon(y, u, z, c, d, h, dt=1.0):
    """
    Predykcja h-krokowa:
    y_hat[n] liczona wyłącznie z danych archiwalnych
    """
    N = len(y)
    y_hat = np.full(N, np.nan)

    for n in range(h, N):
        y_tmp = y[n - h]

        for k in range(h):
            idx = n - h + k
            y_tmp = y_tmp + (dt / c) * (u[idx] + d * z[idx] - d * y_tmp)

        y_hat[n] = y_tmp

    return y_hat


def separate_prediction_plots(t, h_list, y, predictions):
    """
    Tworzy osobne wykresy rzeczywistego przebiegu i predykcji dla każdego h.
    """
    plt.figure(figsize=(12, 6))

    for i, h in enumerate(h_list):
        plt.subplot(math.ceil(len(h_list)/2), 2, i + 1)
        plt.plot(
            t,
            predictions[h],
            linestyle="--",
            label=f"ŷ – predykcja, h={h}"
        )
        plt.plot(t, y, label="y – pomiar", linewidth=2)

        plt.xlabel("n (próbka)")
        plt.ylabel("Temperatura")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.suptitle("Przebieg rzeczywisty i predykcje modelu")
    plt.show()


def combo_prediction_plot(t, h_list, y, predictions):
    """
    Tworzy wspólny wykres rzeczywistego przebiegu i predykcji dla każdego h.
    """
    plt.figure(figsize=(12, 6))

    for h in h_list:
        plt.plot(
            t,
            predictions[h],
            linestyle="--",
            label=f"ŷ – predykcja, h={h}"
        )

    plt.plot(t, y, label="y – pomiar", linewidth=2)
    plt.xlabel("n (próbka)")
    plt.ylabel("Temperatura")
    plt.title("Przebieg rzeczywisty i predykcje modelu")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


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
    predictions[h] = predict_with_horizon(
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
combo_prediction_plot(t, h_list, y, predictions)
# Wykresy osobne predykcji
separate_prediction_plots(t, h_list, y, predictions)

#----------------------------------- CZĘŚĆ (ii) -----------------------------------------------------------

def count_differences(y, h_list, predictions):
    """
    Oblicza różnice między rzeczywistym przebiegiem a predykcjami dla każdego h.
    """
    differences = {}
    for h in h_list:
        y_hat_h = predictions[h]
        diffs = []
        for i in range(len(y)):
            diff = y[i] - y_hat_h[i]
            diffs.append(diff)
        differences[h] = diffs
    return differences


def combo_plot_differences(t, h_list, differences):
    """
    Tworzy wykresy różnic między rzeczywistym przebiegiem a predykcjami dla każdego h.
    """
    plt.figure(figsize=(12, 6))
    for h in h_list:
        plt.plot(
            t,
            differences[h],
            label=f"Różnica dla h={h}"
        )
    plt.xlabel("n (próbka)")
    plt.ylabel("Różnica y - ŷ")
    plt.title("Różnice między rzeczywistym przebiegiem a predykcjami")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def separate_plots_differences(t, h_list, count_differences):
    """
    Tworzy osobne wykresy różnic między rzeczywistym przebiegiem a predykcjami dla każdego h.
    """
    plt.figure(figsize=(12, 6))
    for i, h in enumerate(h_list):
        plt.subplot(math.ceil(len(h_list)/2), 2, i + 1)
        plt.plot(
            t,
            count_differences[h],
            label=f"Różnica dla h={h}"
        )
        plt.xlabel("n (próbka)")
        plt.ylabel("Różnica y - ŷ")
        plt.title(f"Różnice dla h={h}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.suptitle("Różnice między rzeczywistym przebiegiem a predykcjami")
    plt.show()


# ZADANIE 1
differences = count_differences(y, h_list, predictions)
# Wykres zbiorczy różnic
combo_plot_differences(t, h_list, differences)
# Wykresy osobne różnic
separate_plots_differences(t, h_list, differences)

# ZADANIE 2
df = pd.read_excel('L-4/przebieg_zab.xlsx')

# Dane zaburzonego przebiegu
y_zab = df["y (T)"].values.astype(float)
u_zab = df["u (Pg)"].values.astype(float)
z_zab = df["z (Tz)"].values.astype(float)

# Predykcje dla zaburzonego przebiegu
predictions_zab = {}
for h in h_list:
    predictions_zab[h] = predict_with_horizon(
        y=y_zab,
        u=u_zab,
        z=z_zab,
        c=c_hat,
        d=d_hat,
        h=h,
        dt=dt
    )

# różnice dla zaburzonego przebiegu
differences_zab = count_differences(y_zab, h_list, predictions_zab)
t_zab = np.arange(len(y_zab))

# Wykres zbiorczy różnic dla zaburzonego przebiegu
combo_plot_differences(t_zab, h_list, differences_zab)

