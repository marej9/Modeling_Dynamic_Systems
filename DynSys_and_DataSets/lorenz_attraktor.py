import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint

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
initial_state = [50.0, 50.0, 50.5]  # Starting point in phase space
t = np.linspace(0, 50, 10000)    # Time grid for integration

# Numerical integration of the Lorenz equations
trajectory = odeint(lorenz_deriv, initial_state, t, args=(sigma, rho, beta))

# Extract x, y, z components for visualization
x, y, z = trajectory.T

# Save the dataset to a CSV file
data = pd.DataFrame({'Time': t, 'X': x, 'Y': y, 'Z': z})
data.to_csv('lorenz_attractor_dataset_test.csv', index=False)
print("Dataset saved to 'lorenz_attractor_dataset.csv'.")

# Plotting the Lorenz attractor
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, lw=0.5, color='b')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Lorenz Attractor")
plt.show()
