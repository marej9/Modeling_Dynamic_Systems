import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parameters
mu = 0.05          # Drift term
sigma = 0.2        # Volatility term
X0 = 1.0           # Initial value
T = 1.0            # Total time
dt = 0.01          # Time step
N = int(T / dt)    # Number of steps
time = np.linspace(0, T, N)

# Initialize the array for X values
X = np.zeros(N)
X[0] = X0

# Simulate the stochastic process
for t in range(1, N):
    dW = np.sqrt(dt) * np.random.normal()
    X[t] = X[t - 1] + mu * X[t - 1] * dt + sigma * X[t - 1] * dW

# Save the data to a CSV file
data = pd.DataFrame({'Time': time, 'X': X})
data.to_csv('gbm_timeseries.csv', index=False)
print("Time series data saved to 'gbm_timeseries.csv'")

# Plot the time series
plt.plot(time, X, label="GBM Simulation")
plt.xlabel("Time")
plt.ylabel("X")
plt.title("Geometric Brownian Motion Simulation")
plt.legend()
plt.show()