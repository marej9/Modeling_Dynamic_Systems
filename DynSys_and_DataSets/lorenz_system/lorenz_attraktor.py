import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint
import os

# Parameters for the Lorenz system
sigma = 10
rho = 28
beta = 8 / 3

# Define the Lorenz system equations
def lorenz_deriv(state, t, sigma, rho, beta):
    x, y, z = state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return [dx_dt, dy_dt, dz_dt]

# Initial conditions and time span
initial_state = [0.001, 0.001, 0.001]  # Starting point in phase space
t = np.linspace(0, 100, 20000)    # Time grid for integration

# Numerical integration of the Lorenz equations
trajectory = odeint(lorenz_deriv, initial_state, t, args=(sigma, rho, beta))

# Extract x, y, z components for visualization
x, y, z = trajectory.T

# Save the dataset to a CSV file
data = pd.DataFrame({'Time': t, 'X': x, 'Y': y, 'Z': z})
file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lorenz_system_data_5.csv")
data.to_csv(file_path, index=False)
print(f"Dataset saved to {file_path}'.")

# Plotting the Lorenz attractor
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, lw=0.5, color='b')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Lorenz Attractor")
plt.show()


