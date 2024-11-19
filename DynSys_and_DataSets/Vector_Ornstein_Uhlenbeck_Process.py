
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Parameters for the simulation
num_variables = 3   # Number of output variables
T = 10              # Total time
dt = 0.01           # Time step
N = int(T / dt)     # Number of steps
time = np.linspace(0, T, N)

# Mean vector (target values each variable reverts to)
mu = np.array([1.0, 0.5, -0.5])

# Mean reversion matrix (diagonal here, can be customized for interactions)
Theta = np.array([[0.5, 0.0, 0.0],
                  [0.0, 0.3, 0.0],
                  [0.0, 0.0, 0.7]])

# Volatility matrix (introducing some correlation between outputs)
Sigma = np.array([[0.1, 0.05, 0.02],
                  [0.05, 0.2, 0.03],
                  [0.02, 0.03, 0.15]])

# Initialize state matrix to hold all variables over time
X = np.zeros((N, num_variables))
X[0] = np.random.normal(mu, 0.1)  # Initial state

# Simulate the process
for t in range(1, N):
    # Generate correlated noise
    dW = np.random.multivariate_normal(np.zeros(num_variables), dt * np.eye(num_variables))
    # Calculate the next state
    X[t] = X[t - 1] + Theta @ (mu - X[t - 1]) * dt + Sigma @ dW

# Convert data to a DataFrame
data = pd.DataFrame(X, columns=[f"X{i+1}" for i in range(num_variables)])
data['Time'] = time

# Save to CSV file
data.to_csv('multivariate_ou_timeseries.csv', index=False)
print("Time series data saved to 'multivariate_ou_timeseries.csv'")

# Plot the time series for visualization
plt.figure(figsize=(10, 6))
for i in range(num_variables):
    plt.plot(time, X[:, i], label=f"X{i+1}")
plt.xlabel("Time")
plt.ylabel("Value")
plt.title("Multivariate Ornstein-Uhlenbeck Process Simulation")
plt.legend()
plt.show()
