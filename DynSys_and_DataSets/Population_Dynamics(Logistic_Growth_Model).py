import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Parameters for the logistic growth model
r = 0.1          # Growth rate
K = 1000         # Carrying capacity
P0 = 10          # Initial population
T = 100          # Total time for simulation
dt = 0.1         # Time step
t = np.arange(0, T, dt)  # Time array

# Logistic growth model differential equation
def logistic_growth(P, t, r, K):
    dP_dt = r * P * (1 - P / K)
    return dP_dt

# Integrate the differential equation to get population over time
population = odeint(logistic_growth, P0, t, args=(r, K)).flatten()

# Create a DataFrame and save to CSV
data = pd.DataFrame({'Time': t, 'Population': population})
data.to_csv('population_dynamics_logistic_growth.csv', index=False)
print("Dataset saved to 'population_dynamics_logistic_growth.csv'.")

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(t, population, label="Population (P)", color='b')
plt.xlabel("Time")
plt.ylabel("Population")
plt.title("Population Dynamics (Logistic Growth Model)")
plt.legend()
plt.grid()
plt.show()
