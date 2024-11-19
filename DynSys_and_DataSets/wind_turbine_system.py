import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Simulation parameters
T = 100               # Total time (seconds)
dt = 0.1              # Time step
N = int(T / dt)       # Number of time steps
time = np.linspace(0, T, N)

# Wind turbine parameters
J = 5.0               # Moment of inertia (example value)
rho = 1.225           # Air density (kg/m^3)
R = 40                # Rotor radius (meters)
A = np.pi * R**2      # Swept area

# Functions for aerodynamic torque and power coefficient
def power_coefficient(lambda_, beta):
    lambda_i = 1 / (1 / (lambda_ + 0.08 * beta) - 0.035 / (beta**3 + 1))
    return 0.5 * (116 / lambda_i - 0.4 * beta - 5) * np.exp(-21 / lambda_i)

def aerodynamic_torque(omega, v, beta):
    lambda_ = omega * R / v
    Cp = power_coefficient(lambda_, beta)
    return 0.5 * rho * A * v**3 * Cp / omega

# Initialize variables
omega = np.zeros(N)         # Rotor speed
T_m = np.zeros(N)           # Aerodynamic torque
T_g = 0.2                   # Generator torque (constant for simplicity)
wind_speed = 8 + 2 * np.sin(0.1 * time)  # Wind speed with a sinusoidal variation
pitch_angle = 5 + 2 * np.cos(0.05 * time)  # Pitch angle variation

# Run simulation
for t in range(1, N):
    T_m[t] = aerodynamic_torque(omega[t-1], wind_speed[t], pitch_angle[t])
    omega[t] = omega[t-1] + (T_m[t] - T_g) / J * dt

# Prepare the dataset
data = pd.DataFrame({
    'Time': time,
    'Rotor Speed (omega)': omega,
    'Aerodynamic Torque (T_m)': T_m,
    'Wind Speed': wind_speed,
    'Pitch Angle': pitch_angle
})

# Save to CSV file
data.to_csv('wind_turbine_simulation.csv', index=False)
print("Dataset saved to 'wind_turbine_simulation.csv'.")

# Plot the results for visualization
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(time, wind_speed, label="Wind Speed (m/s)")
plt.plot(time, pitch_angle, label="Pitch Angle (degrees)")
plt.ylabel("Wind Speed / Pitch Angle")
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(time, omega, label="Rotor Speed (rad/s)")
plt.ylabel("Rotor Speed")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(time, T_m, label="Aerodynamic Torque (Nm)")
plt.xlabel("Time (s)")
plt.ylabel("Aerodynamic Torque")
plt.legend()

plt.tight_layout()
plt.show()
