import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Parameters
m0 = 5000        # Initial mass of the missile (kg)
dm_dt = 10       # Fuel burn rate (kg/s)
T = 100000       # Thrust force (N)
Cd = 0.5         # Drag coefficient
rho = 1.225      # Air density at sea level (kg/m^3)
A = 1.0          # Cross-sectional area of the missile (m^2)
g = 9.81         # Gravitational acceleration (m/s^2)

# Time array for simulation
t = np.linspace(0, 100, 1000)  # 100 seconds, 1000 time steps

# Define the missile dynamics equations
def missile_dynamics(state, t, T, dm_dt, Cd, rho, A, g, m0):
    x, v = state
    m = m0 - dm_dt * t if (m0 - dm_dt * t) > 0 else 1e-5  # Mass changes over time, ensure mass does not go negative
    D = 0.5 * Cd * rho * A * v**2  # Drag force
    F = T - D - m * g             # Net force acting on the missile
    dv_dt = F / m                 # Acceleration
    dx_dt = v                     # Velocity
    return [dx_dt, dv_dt]

# Initial conditions
initial_state = [0, 0]  # Starting at rest at position 0

# Integrate the equations
trajectory = odeint(missile_dynamics, initial_state, t, args=(T, dm_dt, Cd, rho, A, g, m0))
position, velocity = trajectory.T

# Create a DataFrame and save to CSV
data = pd.DataFrame({'Time': t, 'Position': position, 'Velocity': velocity})
data.to_csv('missile_dynamics_with_changing_mass.csv', index=False)
print("Dataset saved to 'missile_dynamics_with_changing_mass.csv'.")

# Plotting the results
plt.figure(figsize=(12, 6))

# Position plot
plt.subplot(2, 1, 1)
plt.plot(t, position, label="Position", color='b')
plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.title("Missile Dynamics with Changing Mass")
plt.grid()
plt.legend()

# Velocity plot
plt.subplot(2, 1, 2)
plt.plot(t, velocity, label="Velocity", color='r')
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()
