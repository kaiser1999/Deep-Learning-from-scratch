import numpy as np
import pandas as pd
from scratch import Dense
from scratch import Adam
from scratch import SimpleRNN, GRU, LSTM
from scratch import Sequential

#%%
Year = 2018
LAG = 20

df = pd.read_csv(f"Datasets/Bitstamp_BTCUSD_{Year}_minute.csv", header=1)
df.index = df.date
df.drop(["unix", "date", "symbol", "Volume USD"], axis=1, inplace=True)
df = df.iloc[::-1]          # Reverse the order of dates

BTC_vol = df["Volume BTC"].values
df_diff = df.diff()
df_diff["Volume BTC"] = BTC_vol

# Select the last quarter as the training dataset
date_index = pd.to_datetime(df_diff.index)
mask_train = pd.Series(date_index).between(f"{Year}-09-30", f"{Year}-12-30")
df_train = df_diff.loc[mask_train.values]
y_close_train = df.close.loc[mask_train.values]

# Select the first day as the test dataset
mask_test = pd.Series(date_index).between(f"{Year}-12-30", f"{Year}-12-31")
df_test = df_diff.loc[mask_test.values]
# Add LAG number of observations in training dataset to test dataset
df_test = pd.concat((df_train.iloc[-LAG:,:], df_test), axis=0)
y_close_test = df.close[mask_test.values]

#%%
def generate_dataset(df, seq_len):
    X_list, y_list = [], []
    for i in range(len(df.index) - seq_len):
        X_list.append(np.array(df.iloc[i:i+seq_len,:]))
        y_list.append(df.close.values[i+seq_len])
        
    return np.array(X_list), np.array(y_list)

X_train, y_train = generate_dataset(df_train, seq_len=LAG)
X_test, y_test = generate_dataset(df_test, seq_len=LAG)

print(np.mean(y_train))
print(np.mean((y_train - np.mean(y_train))**2))
print(np.mean((y_test - np.mean(y_train))**2))

#%%
np.random.seed(4012)

SimpleRNN_model = Sequential()
SimpleRNN_model.add(SimpleRNN(32, a_func="tanh", return_sequence=True))
SimpleRNN_model.add(SimpleRNN(16, a_func="tanh", return_sequence=False))
SimpleRNN_model.add(Dense(1))
SimpleRNN_model.compile(Adam(1e-3), loss="MSE")
SimpleRNN_model.train(X_train, y_train, bs=64, EPOCHS=20)

SimpleRNN_pred = np.squeeze(SimpleRNN_model.predict(X_test))
print(np.mean((SimpleRNN_pred - y_test)**2))

#%%
np.random.seed(4012)

GRU_model = Sequential()
GRU_model.add(GRU(32, a_func="tanh", return_sequence=True))
GRU_model.add(GRU(16, a_func="tanh", return_sequence=False))
GRU_model.add(Dense(1))
GRU_model.compile(Adam(1e-3), loss="MSE")
GRU_model.train(X_train, y_train, bs=64, EPOCHS=20)

GRU_pred = np.squeeze(GRU_model.predict(X_test))
print(np.mean((GRU_pred - y_test)**2))

#%%
np.random.seed(4012)

LSTM_model = Sequential()
LSTM_model.add(LSTM(32, a_func="tanh", return_sequence=True))
LSTM_model.add(LSTM(16, a_func="tanh", return_sequence=False))
LSTM_model.add(Dense(1))
LSTM_model.compile(Adam(1e-3), loss="MSE")
LSTM_model.train(X_train, y_train, bs=64, EPOCHS=20)

LSTM_pred = np.squeeze(LSTM_model.predict(X_test))
print(np.mean((LSTM_pred - y_test)**2))

#%%
import matplotlib.pyplot as plt

nminute = 500
date_val = df_test.index[LAG:][:nminute]
xticks = pd.to_datetime(date_val).strftime('%H:%M')
plt.rcParams['font.size'] = "20"
for pred, name, c in zip([SimpleRNN_pred, GRU_pred, LSTM_pred],
                         ["SimpleRNN", "GRU", "LSTM"],
                         ["blue", "red", "green"]):
    close = pred + y_close_test
    fig = plt.figure(figsize=(15,10))
    plt.plot(close[:nminute], color=c, linestyle='dashed', 
             label=name, linewidth=4)
    plt.plot(y_close_test[:nminute], color='orange', label='Actual')
    plt.xticks(date_val[:nminute:int(nminute/10)], 
               xticks[:nminute:int(nminute/10)])
    plt.title(f'''Bitcoin Price Prediction from {date_val[0]} 
              to {date_val[-1]}''', fontsize=25)
    plt.ylim(3690, 3770)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"../../Graphs/Bitcoin RNNs_{name}.png", dpi=200)