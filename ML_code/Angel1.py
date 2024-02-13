import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt


class Angel7:
    def __init__(self, C_Name) -> None:
        self.C_name = C_Name
        self.data_frame = pd.read_csv(f"./{self.C_name}.csv")
    
    def fit(self):
        self.data_frame["Date"] = pd.to_datetime(
            self.data_frame.Date, format="%Y-%m-%d"
        )
        self.data_frame.index = self.data_frame["Date"]

        data = self.data_frame.sort_index(ascending=True, axis=0)
        new_data = pd.DataFrame(
            index=range(0, len(self.data_frame)), columns=["Date", "Close"]
        )
        for i in range(0, len(data)):
            new_data["Date"][i] = data["Date"][i]
            new_data["Close"][i] = data["Close"][i]

        new_data.index = new_data.Date
        new_data.drop("Date", axis=1, inplace=True)

        dataset = new_data.values
        train = dataset[0:3300, :]
        valid = dataset[3300:, :]

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)

        x_train, y_train = [], []
        for i in range(60, len(train)):
            x_train.append(scaled_data[i - 60 : i, 0])
            y_train.append(scaled_data[i, 0])
        x_train, y_train = np.array(x_train), np.array(y_train)

        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    
        model = Sequential()
        model.add(
            LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1))
        )
        model.add(LSTM(units=50))
        model.add(Dense(1))
        model.compile(loss="mean_squared_error", optimizer="adam")
        model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)
        inputs = new_data[len(new_data) - len(valid) - 60 :].values
        inputs = inputs.reshape(-1, 1)
        inputs = scaler.transform(inputs)
        
        X_test = []
        for i in range(60,inputs.shape[0]):
            X_test.append(inputs[i-60:i,0])
        X_test = np.array(X_test)
        
        X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
        closing_price = model.predict(X_test)
        closing_price = scaler.inverse_transform(closing_price)
        
        train = new_data[3000:3300]
        valid = new_data[3300:]
        valid.loc[:,'Predictions'] = closing_price
        plt.plot(train.loc[:,'Close'],'r',label='History')
        plt.plot(valid.loc[:,'Predictions'],label = 'Prediction')
        plt.xticks([])
        plt.ylabel('Price in Rs.')
        plt.legend()
        plt.savefig(f"{self.C_name}.png", bbox_inches = 'tight')
        plt.close()
        
        
Companies =['SBIN','HDFCBANK','SUNPHARMA' ]
for company in Companies:
    stock = Angel7(company)
    stock.fit()
    del stock
        
        