# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Model
from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, RepeatVector, TimeDistributed


# %%
# Load the data
dataset_train = pd.read_csv(r'C:\Users\damja\OneDrive\Desktop\Thesis\dataset\GOOG-year.csv')

# Extract relevant columns
data = dataset_train[['Close']]

# Normalize the data
sc = MinMaxScaler(feature_range=(0, 1))
data_scaled = sc.fit_transform(data)

# %%
# Create sequences for training
X_train = []
y_train = []

for i in range(60, len(data_scaled)):
    X_train.append(data_scaled[i-60:i, 0])  # Use only 'Close' column for input
    y_train.append(data_scaled[i, 0])  # Predict the 'Close' column only

X_train, y_train = np.array(X_train), np.array(y_train)

# %%
# Reshape X_train for CNN
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# %%
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')


# %%
# Train the model
model.fit(X_train, y_train, epochs=1500, batch_size=32)

# %%
# Prepare test data
dataset_test = pd.read_csv(r'C:\Users\damja\OneDrive\Desktop\Thesis\dataset\GOOG-year.csv')
real_stock_price = dataset_test['Close'].values

# Extract relevant columns
data_test = dataset_test[['Close']]

# %%
# Normalize the test data
inputs_test = sc.transform(data_test)

# Create sequences for testing
X_test = []

for i in range(60, len(inputs_test)):
    X_test.append(inputs_test[i-60:i, 0])  # Use only 'Close' column for input

X_test = np.array(X_test)

# %%
# Reshape X_test for CNN
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Predict 'Close' column
predicted_close_prices = model.predict(X_test)
predicted_close_prices = sc.inverse_transform(predicted_close_prices)

# %%
# Plotting the results
plt.plot(real_stock_price, color='red', label='Real Google Stock Price (Close)')
plt.plot(predicted_close_prices, color='blue', label='Predicted Google Stock Price (Close)')
plt.title('Google Stock Price Prediction (Close) using CNN')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()


