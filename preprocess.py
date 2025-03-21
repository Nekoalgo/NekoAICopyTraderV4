import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the raw Forex data generated by Alpha Vantage
df = pd.read_csv("data/forex_data.csv")
print("Columns in raw data:", df.columns)

# For Alpha Vantage data, we expect a 'close' column.
if "close" not in df.columns:
    raise Exception("The raw data must have a 'close' column for price data.")

# Use the 'close' column for price values.
prices = df["close"].values.reshape(-1, 1)

# Normalize the prices between 0 and 1.
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)

# Replace the 'close' column with the normalized values.
df["close"] = scaled_prices

# Save the preprocessed data to a new CSV file.
df.to_csv("data/forex_data_preprocessed.csv", index=False)
print("Preprocessing complete. Data saved as 'data/forex_data_preprocessed.csv'.")
