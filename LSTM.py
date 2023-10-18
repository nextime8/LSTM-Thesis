# %%
import pandas as pd

df = pd.read_csv(r'data\NVDA15.csv')

df


# %%
df = df[['Date', 'Close']]

df

df['Date']

# %%
import datetime

def str_to_datetime(s):
  split = s.split('-')
  year, month, day = int(split[0]), int(split[1]), int(split[2])
  return datetime.datetime(year=year, month=month, day=day)

datetime_object = str_to_datetime('1999-12-15')
datetime_object


# %%
df

df['Date'] = df['Date'].apply(str_to_datetime)
df['Date']


# %%
df.index = df.pop('Date')
df

# %%
import matplotlib.pyplot as plt

plt.plot(df.index, df['Close'])

# %%
import numpy as np

def df_to_windowed_df(dataframe, first_date_str, last_date_str, n=3):
  first_date = str_to_datetime(first_date_str)
  last_date  = str_to_datetime(last_date_str)

  target_date = first_date
  
  dates = []
  X, Y = [], []

  last_time = False
  while True:
    df_subset = dataframe.loc[:target_date].tail(n+1)
    
    if len(df_subset) != n+1:
      print(f'Error: Window of size {n} is too large for date {target_date}')
      return

    values = df_subset['Close'].to_numpy()
    x, y = values[:-1], values[-1]

    dates.append(target_date)
    X.append(x)
    Y.append(y)

    next_week = dataframe.loc[target_date:target_date+datetime.timedelta(days=7)]
    next_datetime_str = str(next_week.head(2).tail(1).index.values[0])
    next_date_str = next_datetime_str.split('T')[0]
    year_month_day = next_date_str.split('-')
    year, month, day = year_month_day
    next_date = datetime.datetime(day=int(day), month=int(month), year=int(year))
    
    if last_time:
      break
    
    target_date = next_date

    if target_date == last_date:
      last_time = True
    
  ret_df = pd.DataFrame({})
  ret_df['Target Date'] = dates
  
  X = np.array(X)
  for i in range(0, n):
    X[:, i]
    ret_df[f'Target-{n-i}'] = X[:, i]
  
  ret_df['Target'] = Y

  return ret_df

# %%
# Start day second time around: '2022-10-17'
windowed_df = df_to_windowed_df(df, 
                                '2022-10-17', 
                                '2023-10-16', 
                                n=3)
windowed_df
def windowed_df_to_date_X_y(windowed_dataframe):
  df_as_np = windowed_dataframe.to_numpy()

  dates = df_as_np[:, 0]

  middle_matrix = df_as_np[:, 1:-1]
  X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1))

  Y = df_as_np[:, -1]

  return dates, X.astype(np.float32), Y.astype(np.float32)


# %%
dates, X, y = windowed_df_to_date_X_y(windowed_df)

dates.shape, X.shape, y.shape

# %%
q_80 = int(len(dates) * .8)
q_90 = int(len(dates) * .9)

dates_train, X_train, y_train = dates[:q_80], X[:q_80], y[:q_80]

dates_val, X_val, y_val = dates[q_80:q_90], X[q_80:q_90], y[q_80:q_90]
dates_test, X_test, y_test = dates[q_90:], X[q_90:], y[q_90:]

# %%
plt.plot(dates_train, y_train)
plt.plot(dates_val, y_val)
plt.plot(dates_test, y_test)

plt.legend(['Train', 'Validation', 'Test'])

# %%
# Function to load and preprocess data
def load_and_preprocess_data(filename, start_date, end_date, n=3):
    df = pd.read_csv(filename)
    df = df[['Date', 'Close']]
    df['Date'] = df['Date'].apply(str_to_datetime)
    df.index = df.pop('Date')

    # Create windowed dataframe
    windowed_df = df_to_windowed_df(df, start_date, end_date, n=n)

    # Convert windowed dataframe to date, X, and y
    dates, X, y = windowed_df_to_date_X_y(windowed_df)

    # Split data into train, validation, and test sets
    q_80 = int(len(dates) * 0.8)
    q_90 = int(len(dates) * 0.9)

    dates_train, X_train, y_train = dates[:q_80], X[:q_80], y[:q_80]
    dates_val, X_val, y_val = dates[q_80:q_90], X[q_80:q_90], y[q_80:q_90]
    dates_test, X_test, y_test = dates[q_90:], X[q_90:], y[q_90:]

    return dates_train, X_train, y_train, dates_val, X_val, y_val, dates_test, X_test, y_test

# Load and preprocess data for MSFT.csv
dates_train_msft, X_train_msft, y_train_msft, dates_val_msft, X_val_msft, y_val_msft, dates_test_msft, X_test_msft, y_test_msft = load_and_preprocess_data(r'data\MSFT15.csv', '2022-10-17', '2023-10-16', n=3)

# Load and preprocess data for amd15.csv
dates_train_amd, X_train_amd, y_train_amd, dates_val_amd, X_val_amd, y_val_amd, dates_test_amd, X_test_amd, y_test_amd = load_and_preprocess_data(r'data\AMD15.csv', '2022-10-17', '2023-10-16', n=3)

# Load and preprocess data for NVDA15.csv
dates_train_nvda, X_train_nvda, y_train_nvda, dates_val_nvda, X_val_nvda, y_val_nvda, dates_test_nvda, X_test_nvda, y_test_nvda = load_and_preprocess_data(r'data\NVDA15.csv', '2022-10-17', '2023-10-16', n=3)


# %%
from keras.models import Sequential
from keras.optimizers import Adam
from keras import layers

# Function to create and train the model
def create_and_train_model(X_train, y_train, X_val, y_val, epochs=200):
    model = Sequential([
        layers.Input((3, 1)),
        layers.LSTM(64),
        layers.Dense(32, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics=['mean_absolute_error'])

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs)

    return model

# %%
# Create and train the model for MSFT.csv
model_msft = create_and_train_model(X_train_msft, y_train_msft, X_val_msft, y_val_msft)

# Evaluate the model on MSFT.csv test set
test_predictions_msft = model_msft.predict(X_test_msft).flatten()

# Create and train the model for amd15.csv
model_amd = create_and_train_model(X_train_amd, y_train_amd, X_val_amd, y_val_amd)

# Evaluate the model on amd15.csv test set
test_predictions_amd = model_amd.predict(X_test_amd).flatten()

# Create and train the model for NVDA15.csv
model_nvda = create_and_train_model(X_train_nvda, y_train_nvda, X_val_nvda, y_val_nvda)

# Evaluate the model on NVDA15.csv test set
test_predictions_nvda = model_nvda.predict(X_test_nvda).flatten()


# %%
# Plot results for MSFT.csv
plt.figure(figsize=(12, 6))
plt.plot(dates_test_msft, test_predictions_msft, label='MSFT Predictions')
plt.plot(dates_test_msft, y_test_msft, label='MSFT Observations')
plt.title('MSFT Test Predictions vs Observations')
plt.legend()
plt.show()

# %%
# Plot results for amd15.csv
plt.figure(figsize=(12, 6))
plt.plot(dates_test_amd, test_predictions_amd, label='AMD Predictions')
plt.plot(dates_test_amd, y_test_amd, label='amd15 Observations')
plt.title('AMD Test Predictions vs Observations')
plt.legend()
plt.show()

# %%
# Plot results for NVDA15.csv
plt.figure(figsize=(12, 6))
plt.plot(dates_test_nvda, test_predictions_nvda, label='NVDA Predictions')
plt.plot(dates_test_nvda, y_test_nvda, label='NVDA Observations')
plt.title('NVDA15 Test Predictions vs Observations')
plt.legend()
plt.show()

# %%
train_predictions = model.predict(X_train).flatten()

plt.plot(dates_train, train_predictions)
plt.plot(dates_train, y_train)
plt.legend(['Training Predictions', 'Training Observations'])


# %%
val_predictions = model.predict(X_val).flatten()

plt.plot(dates_val, val_predictions)
plt.plot(dates_val, y_val)
plt.legend(['Validation Predictions', 'Validation Observations'])

# %%
test_predictions = model.predict(X_test).flatten()

plt.plot(dates_test, test_predictions)
plt.plot(dates_test, y_test)
plt.legend(['Testing Predictions', 'Testing Observations'])


