import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Parameter für den Van-der-Pol-Oszillator
params_van_der_pol_optimal_construction = (2, 2, 5, 0.8)  # (p_a, p_mu, p_b, p_c)

def make_2d_ct_van_der_pol(p_a, p_mu, p_b, p_c):
    def _contfcn(x):
        x0 = x[:, 0]
        x1 = x[:, 1]
        # Dynamik des Van-der-Pol-Oszillators
        y0 = p_a * x1
        y1 = p_mu * x1 * (1. - p_b * x0**2) - p_c * x0
        return np.stack([y0, y1], axis=1)
    return _contfcn

# Simulation des Systems
def simulate(fcn, x0, timesteps, dt):
    trajectory = [x0]
    x = x0
    for step in range(timesteps):
        dx = fcn(np.array([x]))[0]  # Dynamik für aktuellen Zustand
        x = x + dx * dt
        trajectory.append(x)
    return np.array(trajectory)

# Hyperparameter
initial_state = [0.0001, 0.0001]  # Startzustand im Phasenraum (x0, x1)
simulation_time = 1000        # Gesamtdauer der Simulation in Sekunden
time_steps = 20000          # Anzahl der Zeitschritte
t = np.linspace(0, simulation_time, time_steps)  # Zeitgitter für Integration
dt = t[1] - t[0]            # Zeitschrittgröße berechnen

print(f"Zeitschrittgröße: {dt:.6f} Sekunden")

# Parameter und Funktion erstellen
p_a, p_mu, p_b, p_c = params_van_der_pol_optimal_construction
van_der_pol_fcn = make_2d_ct_van_der_pol(p_a, p_mu, p_b, p_c)

# Simulation mit den neuen Parametern
trajectory = simulate(van_der_pol_fcn, np.array(initial_state), len(t) - 1, dt)

# CSV-Export
data = np.hstack([t.reshape(-1, 1), trajectory])   # Zeit und Zustände kombinieren
columns = ["Time", "x0", "x1"]
df = pd.DataFrame(data, columns=columns)

# Speichern in CSV
csv_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),"van_der_pol_data_5.csv")
df.to_csv(csv_file, index=False)
print(f"Datensatz wurde als '{csv_file}' gespeichert.")

# Plot der Trajektorie
plt.figure(figsize=(6, 6))
plt.plot(trajectory[:, 0], trajectory[:, 1], label="Van der Pol Oscillator")
plt.title("Van der Pol Oscillator - Phase Space")
plt.xlabel("x0")
plt.ylabel("x1")
plt.axis('equal')
plt.legend()
plt.grid(True)
plt.show()
