import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

df = pd.read_excel('L-4/przebieg_norm.xlsx')

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


# ZADANIE 2
y = df["y (T)"].values.astype(float)
u = df["u (Pg)"].values.astype(float)
z = df["z (Tz)"].values.astype(float)

dt = 1.0


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

# Wykresy osobne dla każdego h
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

# Wykres zbiorczy
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