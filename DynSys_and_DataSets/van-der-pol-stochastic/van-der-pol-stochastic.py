import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import os

# Parameter für den Van-der-Pol-Oszillator
p_a, p_mu, p_b, p_c = 2, 2, 5, 0.8

# Definition der Van-der-Pol-Gleichungen mit stochastischer Komponente
def van_der_pol_deriv(state, p_a, p_mu, p_b, p_c, noise_std):
    x0, x1 = state
    dx0_dt = p_a * x1 + noise_std * np.random.randn()
    dx1_dt = p_mu * x1 * (1. - p_b * x0**2) - p_c * x0 + noise_std * np.random.randn()
    return np.array([dx0_dt, dx1_dt])

# Funktion zur Generierung von Van-der-Pol-Daten basierend auf Sampling-Parametern und Initialwerten
def generate_van_der_pol_data(num_points, sampling_rate, initial_state, period, noise_std):
    """
    Generiert Van-der-Pol-Daten basierend auf Sampling-Parametern und Initialwerten.

    Parameter:
    - num_points: Gesamtanzahl der Datenpunkte
    - sampling_rate: Anzahl der Punkte pro Periode
    - initial_state: Liste der Anfangswerte [x0, x1]
    - period: Charakteristische Periode des Van-der-Pol-Oszillators (Beispielwert)
    - noise_std: Standardabweichung des Rauschens

    Rückgabe:
    - DataFrame mit den Van-der-Pol-Daten (Time, x0, x1)
    """
    
    # Berechnung der Gesamtdauer der Simulation basierend auf num_points und sampling_rate
    total_time = num_points * period / sampling_rate
    dt = total_time / num_points
    
    # Initialisierung der Trajektorie
    trajectory = np.zeros((num_points, 2))
    trajectory[0] = initial_state
    
    # Numerische Integration der Van-der-Pol-Gleichungen mit stochastischer Komponente
    for i in range(1, num_points):
        trajectory[i] = trajectory[i-1] + van_der_pol_deriv(trajectory[i-1], p_a, p_mu, p_b, p_c, noise_std) * dt

    # Zeitraster für die Integration
    t = np.linspace(0, total_time, num_points)

    # Extrahieren der x0- und x1-Komponenten zur Visualisierung
    x0, x1 = trajectory.T

    # Erstellen eines DataFrames mit den Ergebnissen
    data = pd.DataFrame({'Time': t, 'x0': x0, 'x1': x1})
    
    return data

# Hyperparameter
num_points = 60000  # Gesamtanzahl der Datenpunkte
sampling_rate = 12  # Anzahl der Punkte pro Periode
initial_state = [1.0, 1.0]  # Angepasste Anfangswerte [x0, x1]
period = 1  # Charakteristische Periode des Van-der-Pol-Oszillators
signal_to_noise_ratio_db = 20  # Signal-Rausch-Verhältnis in dB

# Umrechnung von dB in lineares Verhältnis
signal_to_noise_ratio_linear = 10 ** (signal_to_noise_ratio_db / 10)

# Berechnung der Standardabweichung des Rauschens basierend auf dem Signal-Rausch-Verhältnis
noise_std = np.sqrt(2 / signal_to_noise_ratio_linear)

# Generierung der Van-der-Pol-Daten mit stochastischer Komponente
van_der_pol_data = generate_van_der_pol_data(num_points, sampling_rate, initial_state, period, noise_std)

# Speichern des Datensatzes in einer CSV-Datei
file_path = "van_der_pol_stochastic_data.csv"
van_der_pol_data.to_csv(file_path, index=False)
print(f"Dataset saved to {file_path}'")

# Plotten der Van-der-Pol-Trajektorie mit stochastischer Komponente
plt.figure(figsize=(10, 8))
plt.plot(van_der_pol_data['x0'], van_der_pol_data['x1'], lw=0.5, color='b')
plt.xlabel("x0")
plt.ylabel("x1")
plt.title("Van der Pol Oscillator with Stochastic Component (25dB SNR) - Phase Space")
plt.grid(True)
plt.show()