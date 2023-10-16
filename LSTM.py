# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense

# %%
# Load the dataset
dataset = pd.read_csv(r'C:\Users\damja\OneDrive\Desktop\Thesis\dataset\GOOG-year.csv')
training_set = dataset.iloc[:, 1:2].values

# %%
dataset.head()

# %%
# Feature scaling
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

# %%
# Create sequences for training
X_train = []
y_train = []

for i in range(60, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

# Reshape the data for LSTM
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
y_train = y_train.reshape(-1, 1)


# %%
# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=25, activation='relu'))  # Additional dense layer
model.add(Dense(units=1))  # Output layer
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model with early stopping
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)


# %%
# Train the model
model.fit(X_train, y_train, epochs=150, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

# %%
# Prepare the test data
dataset_test = pd.read_csv(r'C:\Users\damja\OneDrive\Desktop\Thesis\dataset\GOOG-year.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# %%
dataset_total = pd.concat((dataset['Open'], dataset_test['Open']), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)

X_test = []

for i in range(60, len(inputs)):
    X_test.append(inputs[i-60:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))



# %%
# Make predictions
predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# %%
from sklearn.metrics import mean_absolute_error
# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(real_stock_price, predicted_stock_price)
print(f'Mean Absolute Percentage Error (MAPE): {mae:.2f}%')

# %%
# Plot the results
plt.figure(figsize=(12, 6))

# Plotting the actual stock prices
plt.plot(real_stock_price, color='blue', label='Actual Stock Price')

# Plotting the predicted stock prices
plt.plot(predicted_stock_price, color='red', label='Predicted Stock Price')

plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()


