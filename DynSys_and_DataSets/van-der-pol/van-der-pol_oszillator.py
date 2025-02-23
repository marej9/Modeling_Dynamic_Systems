import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.integrate import odeint

# Parameter für den Van-der-Pol-Oszillator
p_a, p_mu, p_b, p_c = 2, 2, 5, 0.8

# Definition der Van-der-Pol-Gleichungen
def van_der_pol_deriv(state, t, p_a, p_mu, p_b, p_c):
    x0, x1 = state
    dx0_dt = p_a * x1
    dx1_dt = p_mu * x1 * (1. - p_b * x0**2) - p_c * x0
    return [dx0_dt, dx1_dt]

# Funktion zur Generierung von Van-der-Pol-Daten basierend auf Sampling-Parametern und Initialwerten
def generate_van_der_pol_data(num_points, sampling_rate, initial_state, period):
    """
    Generiert Van-der-Pol-Daten basierend auf Sampling-Parametern und Initialwerten.

    Parameter:
    - num_points: Gesamtanzahl der Datenpunkte
    - sampling_rate: Anzahl der Punkte pro Periode
    - initial_state: Liste der Anfangswerte [x0, x1]
    - period: Charakteristische Periode des Van-der-Pol-Oszillators (Beispielwert)

    Rückgabe:
    - DataFrame mit den Van-der-Pol-Daten (Time, x0, x1)
    """
    
    # Berechnung der Gesamtdauer der Simulation basierend auf num_points und sampling_rate
    total_time = num_points * period / sampling_rate
    
    # Zeitraster für die Integration
    t = np.linspace(0, total_time, num_points)

    # Numerische Integration der Van-der-Pol-Gleichungen
    trajectory = odeint(van_der_pol_deriv, initial_state, t, args=(p_a, p_mu, p_b, p_c))

    # Extrahieren der x0- und x1-Komponenten zur Visualisierung
    x0, x1 = trajectory.T

    # Erstellen eines DataFrames mit den Ergebnissen
    data = pd.DataFrame({'Time': t, 'x0': x0, 'x1': x1})
    
    return data

# Hyperparameter
num_points = 50000  # Gesamtanzahl der Datenpunkte
sampling_rate = 8  # Anzahl der Punkte pro Periode
initial_state = [0.0001, 0.0001]  # Anfangswerte [x0, x1]
period = 1  # Charakteristische Periode des Van-der-Pol-Oszillators

# Generierung der Van-der-Pol-Daten
van_der_pol_data = generate_van_der_pol_data(num_points, sampling_rate, initial_state, period)

# Speichern des Datensatzes in einer CSV-Datei
file_path = "van_der_pol_data.csv"
van_der_pol_data.to_csv(file_path, index=False)
print(f"Dataset saved to {file_path}'.")

# Plotten der Van-der-Pol-Trajektorie
plt.figure(figsize=(10, 8))
plt.plot(van_der_pol_data['x0'], van_der_pol_data['x1'], lw=0.5, color='b')
plt.xlabel("x0")
plt.ylabel("x1")
plt.title("Van der Pol Oscillator - Phase Space")
plt.grid(True)
plt.show()