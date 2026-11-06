
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
print(data.drop(columns=["date"]).describe())

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


# ---------------------------------
# DATA PREPROCESSING
# ---------------------------------

# Split data chronologically (80% training, 20% testing)
split_index = int(len(data) * 0.8)

train_data = data.iloc[:split_index].copy()
test_data = data.iloc[split_index:].copy()

# Use date as index for time-based processing
train_data = train_data.set_index("date")
test_data = test_data.set_index("date")

# Select weather features
weather_columns = train_data.columns

# Calculate medians using training data only
train_medians = train_data.median()

# Fill missing values using previous observations
train_data = train_data.ffill()

# Fill remaining missing values using training medians
train_data = train_data.fillna(train_medians)

# Fill missing test values using previous observations
test_data = test_data.ffill()

# Use training medians for any remaining missing test values
test_data = test_data.fillna(train_medians)

# Check missing values
print("\nMissing values after preprocessing:")
print("Training:", train_data.isnull().sum().sum())
print("Testing:", test_data.isnull().sum().sum())

print("\nTraining shape:", train_data.shape)
print("Testing shape:", test_data.shape)

print("\nPreprocessing completed successfully!")
