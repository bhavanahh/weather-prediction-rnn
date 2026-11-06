
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("data/london_weather.csv")

# Display basic information
print("First 5 rows:")
print(data.head())

print("\nDataset Shape:")
print(data.shape)

print("\nColumn Information:")
print(data.info())

print("\nMissing Values:")
print(data.isnull().sum())

print("\nDuplicate Rows:")
print(data.duplicated().sum())

# Convert date column
data["date"] = pd.to_datetime(
    data["date"].astype(str),
    format="%Y%m%d"
)

# Sort data by date
data = data.sort_values("date")

# Statistical summary
print("\nStatistical Summary:")
print(data.describe())

# Plot mean temperature over time
plt.figure(figsize=(12, 5))
plt.plot(data["date"], data["mean_temp"])

plt.title("London Mean Temperature Over Time")
plt.xlabel("Date")
plt.ylabel("Mean Temperature (°C)")
plt.grid(True)
plt.tight_layout()

plt.savefig("results/temperature_over_time.png")
plt.show()
