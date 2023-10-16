# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import GRU, Dense, Dropout

# %%
# Load the data
dataset = pd.read_csv(r'C:\Users\damja\OneDrive\Desktop\Thesis\dataset\GOOG-year.csv')

# Extract relevant columns
data = dataset[['Close']]

# Normalize the data
sc = MinMaxScaler(feature_range=(0, 1))
data_scaled = sc.fit_transform(data)

# %%
# Split the data into training (80%) and testing (20%) sets
train_size = int(len(data_scaled) * 0.8)
train_data, test_data = data_scaled[:train_size], data_scaled[train_size:]

# Create sequences for training
X_train, y_train = [], []

for i in range(60, len(train_data)):
    X_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

# Reshape X_train for GRU
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# %%
# Build the GRU model
model = Sequential()
model.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(GRU(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(GRU(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(GRU(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# %%
# Train the model
model.fit(X_train, y_train, epochs=150, batch_size=32)



# %%
# Create sequences for testing
X_test, y_test = [], []

for i in range(60, len(test_data)):
    X_test.append(test_data[i-60:i, 0])
    y_test.append(test_data[i, 0])

X_test, y_test = np.array(X_test), np.array(y_test)

# %%
# Check if X_test is not empty before reshaping
if X_test.shape[0] > 0:
    # Reshape X_test for GRU
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Evaluate the model on the test set
    loss = model.evaluate(X_test, y_test)
    print(f'Mean Squared Error on Test Set: {loss}')


# %%
# Predict on the entire dataset
X_full = []

for i in range(60, len(data_scaled)):
    X_full.append(data_scaled[i-60:i, 0])

X_full = np.array(X_full)


# %%
# Check if X_full is not empty before reshaping
if X_full.shape[0] > 0:
    # Reshape X_full for GRU
    X_full = np.reshape(X_full, (X_full.shape[0], X_full.shape[1], 1))

    # Predict 'Close' column
    predicted_close_prices = model.predict(X_full)
    predicted_close_prices = sc.inverse_transform(predicted_close_prices)

    # Plotting the results
    plt.plot(data.values, color='red', label='Real Google Stock Price (Close)')
    plt.plot(predicted_close_prices, color='blue', label='Predicted Google Stock Price (Close)')
    plt.title('Google Stock Price Prediction (Close) using GRU')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()


