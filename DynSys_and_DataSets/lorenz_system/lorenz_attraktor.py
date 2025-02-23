import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint
import os

# Konstanten für das Lorenz-System
sigma = 10
rho = 28
beta = 8 / 3

# Definition der Lorenz-Gleichungen
def lorenz_deriv(state, t, sigma, rho, beta):
    x, y, z = state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return [dx_dt, dy_dt, dz_dt]

# Funktion zur Generierung von Lorenz-System-Daten basierend auf Sampling-Parametern und Initialwerten
def generate_lorenz_data(num_points, sampling_rate, initial_state, period):
    """
    Generiert Lorenz-System-Daten basierend auf Sampling-Parametern und Initialwerten.

    Parameter:
    - num_points: Gesamtanzahl der Datenpunkte
    - sampling_rate: Anzahl der Punkte pro Periode
    - initial_state: Liste der Anfangswerte [x0, y0, z0]
    - period: Charakteristische Periode des Lorenz-Systems (Beispielwert)

    Rückgabe:
    - DataFrame mit den Lorenz-System-Daten (Time, X, Y, Z)
    """
    
    # Berechnung der Gesamtdauer der Simulation basierend auf num_points und sampling_rate
    total_time = num_points * period / sampling_rate
    
    # Zeitraster für die Integration
    t = np.linspace(0, total_time, num_points)

    # Numerische Integration der Lorenz-Gleichungen
    trajectory = odeint(lorenz_deriv, initial_state, t, args=(sigma, rho, beta))

    # Extrahieren der x-, y-, z-Komponenten zur Visualisierung
    x, y, z = trajectory.T

    # Erstellen eines DataFrames mit den Ergebnissen
    data = pd.DataFrame({'Time': t, 'X': x, 'Y': y, 'Z': z})
    
    return data

num_points = 50000  # Gesamtanzahl der Datenpunkte
sampling_rate = 30  # Anzahl der Punkte pro Periode
initial_state = [0.1, 0.1, 0.1]  # Anfangswerte [x0, y0, z0]
period = 1  # Charakteristische Periode des Lorenz-Systems

# Generierung der Lorenz-System-Daten
lorenz_data = generate_lorenz_data(num_points, sampling_rate, initial_state, period)

# Speichern des Datensatzes in einer CSV-Datei
file_path = "lorenz_system_data.csv"
lorenz_data.to_csv(file_path, index=False)
print(f"Dataset saved to {file_path}'.")

# Plotten des Lorenz-Attraktors
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(lorenz_data['X'], lorenz_data['Y'], lorenz_data['Z'], lw=0.5, color='b')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Lorenz Attractor")
plt.show()