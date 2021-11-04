import math
import numpy as np
import pandas as pd
import pandas_ta as ta
import tensorflow as tf

from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM as keras_lstm
from .base_inducer import BaseInducer


class LSTM(BaseInducer):
    PREDICTION_DAYS = 60

    def __init__(self, model_save_path):
        super().__init__(model_save_path)
        self.scaler = StandardScaler()

        """
        build_train_test_data
        """
        self.n_future = 1  # Number of days we want to look into the future based on the past days.
        self.n_past = 10  # Number of past days we want to use to predict the future.

        """
        build_model/train_model
        """
        self.l2_units = 32
        self.lstm_input_shape1 = None
        self.lstm_input_shape2 = None
        self.dense_input_shape = None
        self.repeat_df_for_training = None


    def save_model(self, model):
        model.save(self.model_save_path)


    def load_model(self):
        return load_model(self.model_save_path)


    def get_data(self, stock_id, date_start, date_end):
        res = self.frontend.get_price_histories(stock_id, date_start, date_end)
        if res.get('status') == 'error':
            raise RuntimeError(df.get('reason'))

        CustomStrategy = ta.Strategy(
            name="FYP321 Multivariate Strategy #1",
            description="SMA 50,200, BBANDS, RSI, MACD and Volume SMA 20",
            ta=[
                {"kind": "bbands", "length": 20},
                {"kind": "sma", "length": 5, "col_names": ("sma5")},
                {"kind": "sma", "length": 8},
                {"kind": "sma", "length": 10},
                {"kind": "wma", "length": 5, "col_names": ("wma5")},
                {"kind": "wma", "length": 8},
                {"kind": "wma", "length": 10},
                {"kind": "macd"},
                {"kind": "rsi"},
                {"kind": "roc"},
                {"kind": "cci"},
                {"kind": "atr"},
                {"kind": "ad"},
            ]
        )

        df = pd.json_normalize(res.get('price_histories'))
        df.ta.strategy(CustomStrategy)
        df = df.fillna(0)

        train_dates = pd.to_datetime(df['date'])
        df = df.drop(columns=['open', 'high', 'low', 'volume', 'change', 'percent_change'])
        cols = list(df)[3:23]
        df_for_training = df[cols].astype(float)
        scaled_data = self.scaler.fit_transform(df_for_training)

        return [train_dates, df_for_training, scaled_data]


    def build_train_test_data(self, raw_data, dummy_arg):

        """
        Formatted training data will be populated into these lists
        """

        df_for_training = raw_data[1]
        df_for_training_scaled = raw_data[2]
        trainX = []
        trainY = []

        """
        Reformat input data into a shape:(n_samples x timesteps x n_features)
        """
        for i in range(self.n_past, len(df_for_training_scaled) - self.n_future + 1):
            trainX.append(df_for_training_scaled[i - self.n_past:i, 0:df_for_training.shape[1]])
            trainY.append(df_for_training_scaled[i + self.n_future - 1:i + self.n_future, 0])
        trainX = np.array(trainX)
        trainY = np.array(trainY)
        self.lstm_input_shape1 = trainX.shape[1]
        self.lstm_input_shape2 = trainX.shape[2]
        self.dense_input_shape = trainY.shape[1]
        self.repeat_df_for_training = df_for_training.shape[1]

        return [trainX, trainY]


    def build_model(self, model_params):
        build_args = model_params['build_args']
        model = Sequential()
        model.add(
            keras_lstm(build_args.get('units', 64), build_args.get('activation', "relu"),
                       input_shape=(self.lstm_input_shape1, self.lstm_input_shape2),
                       return_sequences=True))
        model.add(keras_lstm(self.l2_units, build_args.get('activation', "relu"), return_sequences=False))
        model.add(Dropout(build_args.get('dropout', 0.2)))
        model.add(Dense(self.dense_input_shape))

        model.compile(optimizer='adam', loss='mean_squared_error')
        return model


    def train_model(self, model, model_params, datasets):
        build_args = model_params['build_args']
        history = model.fit(
            datasets[0], datasets[1],
            epochs=build_args.get('epoch', 40),
            batch_size=build_args.get('batch_size', 64),
            validation_split=model_params.get('train_test_percent', 0.1),
            verbose=0
        )
        rmse = math.sqrt(history.history['loss'][-1])

        return {
            'model': model,
            'rmse': rmse,
            'parameters': model_params["build_args"]
        }


    def build_prediction_data(self, dates, data_df, scaled_data):
        """
        The Library will extract business days in the US only!
        """
        last_input_date = list(dates)[-1].date()
        forecast_start_date = last_input_date + timedelta(days = 1)
        us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
        forecast_dates = pd.date_range(
            forecast_start_date, periods=LSTM.PREDICTION_DAYS, freq=us_bd
        ).tolist()

        features = []
        for i in range(self.n_past, len(scaled_data) - self.n_future + 1):
            features.append(scaled_data[i - self.n_past:i, 0:data_df.shape[1]])
        features = np.array(features)

        return [ forecast_dates, features[-LSTM.PREDICTION_DAYS:] ]


    def get_prediction(self, model, forecast_dates, input_features):
        prediction_unscaled = model.predict(input_features)
        prediction_copies = np.repeat(prediction_unscaled, self.repeat_df_for_training, axis=-1)
        prediction_actual = self.scaler.inverse_transform(prediction_copies)[:, 0]

        forecast_dates = [ts.date() for ts in forecast_dates]   # Convert timestamp to date
        result_df = pd.DataFrame({
            'date': np.array(forecast_dates),
            'close': prediction_actual
        })
        result_np = np.array(result_df.iloc[[0, 9, 59], :])

        return {
            'st_date': datetime.strftime(result_np[0][0], "%Y-%m-%d"),
            'st_exp_price': result_np[0][1],
            'mt_date': datetime.strftime(result_np[1][0], "%Y-%m-%d"),
            'mt_exp_price': result_np[1][1],
            'lt_date': datetime.strftime(result_np[2][0], "%Y-%m-%d"),
            'lt_exp_price': result_np[2][1],
        }
