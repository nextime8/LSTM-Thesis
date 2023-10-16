# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dropout, Dense


# %%
# Load the data
dataset_train = pd.read_csv(r'C:\Users\damja\OneDrive\Desktop\Thesis\dataset\GOOG-year.csv')
training_set = dataset_train.iloc[:, 1:2].values

# %%
# Normalize the data
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

# %%
# Create sequences for training
X_train = []
y_train = []

for i in range(60, min(2035, len(training_set_scaled))):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

# %%
# Reshape X_train for CNN
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# %%
# Build the CNN model
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# %%
# Train the model
model.fit(X_train, y_train, epochs=150, batch_size=32)

# %%
# Prepare test data
dataset_test = pd.read_csv(r'C:\Users\damja\OneDrive\Desktop\Thesis\dataset\GOOG-year.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)

# %%
# Create sequences for testing
X_test = []

for i in range(60, 280):
    X_test.append(inputs[i-60:i, 0])

X_test = np.array(X_test)

# Reshape X_test for CNN
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# %%
# Predict stock prices
predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# %%
# Plotting the results
plt.plot(real_stock_price, color='red', label='Real Google Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction using CNN')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()


