import math

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential

DATA_PATH = "data/"  # change the endpoint to database
TOTAL_TIMESTEPS = 100


class PricePredictLSTM:
    model = None
    df_train = None
    df_test = None

    def __init__(self, params, tid, sid, date_s, date_e):
        self.params = params
        self.tid = tid
        self.sid = sid
        self.date_s = date_s
        self.date_e = date_e

        if self.model is None:
            self.load_model()

    def train(self, filename):
        if self.model is None:
            self.load_model()

        if self.df_train is None:
            self.load_data(filename)

        x_train, y_train = self.get_matrix(self.df_train, TOTAL_TIMESTEPS)
        x_test, y_test = self.get_matrix(self.df_test, TOTAL_TIMESTEPS)

        self.model.fit(
            x_train,
            y_train,
            validation_data=(x_test, y_test),
            epochs=100,
            batch_size=64,
            verbose=1
        )

    def test(self, filename):

        if self.model is None:
            self.load_model()

        if self.df_test is None:
            self.load_data(filename)

        x_test, y_test = self.get_matrix(self.df_test, TOTAL_TIMESTEPS)
        test_predict = self.model.predict(x_test)

        scaler = MinMaxScaler(feature_range=(0, 1))
        test_predict = scaler.inverse_transform(test_predict)

        mse = math.sqrt(mean_squared_error(y_test, test_predict))
        print("MSE: ", mse)

        # todo: save the result into database

    def predict(self, x_input, duration: int):
        """

        :param x_input: the input data
        :param duration: the number of days to predict
        :return: an array with all the predicted prices within the duration
        """

        if self.model is None:
            self.load_model()

        # pre-processing
        x_input = x_input.reset_index()['Close'].reshape(1, -1)
        temp_input = list(x_input)
        temp_input = temp_input[0].tolist()

        # 14 days prediction
        output = []
        n_steps = 628
        i = 0

        while i < duration:

            if len(temp_input) > 100:
                x_input = np.array(temp_input[1:])
                x_input = x_input.reshape(1, -1)
                x_input = x_input.reshape((1, n_steps, 1))

                out = self.model.predict(x_input, verbose=0)
                temp_input.extend(out[0].tolist())
                temp_input = temp_input[1:]
                output.extend(out.tolist())

            else:
                x_input = x_input.reshape((1, n_steps, 1))

                out = self.model.predict(x_input, verbose=0)
                temp_input.extend(out[0].tolist())
                output.extend(out.tolist())

            i = i + 1

        output = np.reshape(output, (-1))
        return output

    @staticmethod
    def get_matrix(dataset, total_timesteps=1):
        x, y = [], []
        for i in range(len(dataset) - total_timesteps - 1):
            a = dataset[i:(i + total_timesteps), 0]
            x.append(a)
            y.append(dataset[i + total_timesteps, 0])
        return np.array(x), np.array(y)

    def load_model(self):
        self.model = Sequential()
        self.model.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))
        self.model.add(LSTM(50, return_sequences=True))
        self.model.add(LSTM(50))
        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer='adam')

    def load_data(self, filename):
        # todo: get the data from database

        # get the data from csv file
        path = DATA_PATH + filename
        df = pd.read_csv(path)
        df = df.reset_index()['Close']

        # Normalization
        scaler = MinMaxScaler(feature_range=(0, 1))
        df = scaler.fit_transform(np.array(df).reshape(-1, 1))

        # Split the dataset into training set and test set
        df_size = len(df)
        train_size = int(df_size * 0.65)
        self.df_train, self.df_test = df[0:train_size, :], df[train_size:df_size, :1]
