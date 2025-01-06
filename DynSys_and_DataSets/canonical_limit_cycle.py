import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parameter für den Limit Cycle
params_limit_cycle_ct_ubox = (0.5, 0.3)  # (p_a, p_b)

def make_2d_ct_limit_cycle(p_a, p_b):
    def _contfcn(x):
        x0 = x[:, 0]
        x1 = x[:, 1]
        # Dynamik des Systems
        y0 = -p_a * x1 + x0 * (p_b - x0**2 - x1**2)
        y1 = p_a * x0 + x1 * (p_b - x0**2 - x1**2)
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
initial_state = [1.0, 1.0]  # Startzustand im Phasenraum (x0, x1)
simulation_time = 100        # Gesamtdauer der Simulation in Sekunden
time_steps = 20000          # Anzahl der Zeitschritte
t = np.linspace(0, simulation_time, time_steps)  # Zeitgitter für Integration
dt = t[1] - t[0]            # Zeitschrittgröße berechnen

print(f"Zeitschrittgröße: {dt:.6f} Sekunden")

# Parameter und Funktion erstellen
p_a, p_b = params_limit_cycle_ct_ubox
limit_cycle_fcn = make_2d_ct_limit_cycle(p_a, p_b)

# Simulation mit den neuen Parametern
trajectory = simulate(limit_cycle_fcn, np.array(initial_state), len(t) - 1, dt)

# CSV-Export
data = np.hstack([t.reshape(-1, 1), trajectory])   # Zeit und Zustände kombinieren
columns = ["Time", "x0", "x1"]
df = pd.DataFrame(data, columns=columns)

# Speichern in CSV
csv_file = "limit_cycle_data.csv"
df.to_csv(csv_file, index=False)
print(f"Datensatz wurde als '{csv_file}' gespeichert.")

# Plot der Trajektorie
plt.figure(figsize=(6, 6))
plt.plot(trajectory[:, 0], trajectory[:, 1], label="Limit Cycle")
plt.title("Canonical Limit Cycle")
plt.xlabel("x0")
plt.ylabel("x1")
plt.axis('equal')
plt.legend()
plt.grid(True)
plt.show()
