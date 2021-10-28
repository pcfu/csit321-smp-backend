import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from .base_inducer import BaseInducer
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from sklearn.preprocessing import StandardScaler
import pandas_ta as ta
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM as keras_lstm
import math


class LSTM(BaseInducer):

    def __init__(self, training_id):

        model_save_path = f'trained_models/{training_id}'
        super().__init__(model_save_path)

        """
        build_train_test_data
        """
        self.n_future = 1  # Number of days we want to look into the future based on the past days.
        self.n_past = 10  # Number of past days we want to use to predict the future.

        """
        build_model/train_model
        """
        self.UNITS_2 = 32
        self.VERBOSE = 1

        self.lstm_input_shape1 = None
        self.lstm_input_shape2 = None
        self.dense_input_shape = None
        self.for_model_prediction = None
        self.repeat_df_for_training = None

        """
        get_prediction
        """
        self.PREDICTION_DAYS = 60  # prediction for 3 months
        self.scaler = StandardScaler()

    def get_data(self, stock_id, date_start, date_end):

        params = [stock_id, date_start, date_end]
        df = self.frontend.get_price_histories(*params)

        if df.get('status') == 'error':
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
        df = pd.read_csv('GSK.csv')
        df.ta.strategy(CustomStrategy)
        df = df.fillna(0)

        train_dates = pd.to_datetime(df['Date'])
        df = df.drop(columns=['Volume', 'Open', 'High', 'Low', 'Adj Close'])
        cols = list(df)[1:21]
        df_for_training = df[cols].astype(float)

        # self.scaler = scaler.fit(self.df_for_training)
        df_for_training_scaled = self.scaler.fit_transform(df_for_training)

        raw_data = [train_dates, df_for_training, df_for_training_scaled]

        return raw_data

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
        self.for_model_prediction = trainX[-self.PREDICTION_DAYS:]
        self.repeat_df_for_training = df_for_training.shape[1]

        return [trainX, trainY]

    def build_model(self, model_params):

        build_args = model_params['build_args']
        model = Sequential()
        model.add(
            keras_lstm(build_args.get('units', 64), build_args.get('activation', "relu"),
                       input_shape=(self.lstm_input_shape1, self.lstm_input_shape2),
                       return_sequences=True))
        model.add(keras_lstm(self.UNITS_2, build_args.get('activation', "relu"), return_sequences=False))
        model.add(Dropout(build_args.get('dropout', 0.2)))
        model.add(Dense(self.dense_input_shape))

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.summary()

        return model

    def train_model(self, model, model_params, datasets):

        build_args = model_params['build_args']
        history = model.fit(datasets[0], datasets[1], epochs=build_args.get('epoch', 40),
                            batch_size=build_args.get('batch_size', 64),
                            validation_split=model_params.get('train_test_percent', 0.1), verbose=self.VERBOSE)

        mse = history.history['loss'][-1]
        rmse = math.sqrt(mse)

        return {'model': model,
                'rmse': rmse,
                'parameters': model_params["build_args"]
                }

    def build_prediction_data(self, raw_data):

        train_dates = raw_data[0]
        """
        The Library will extract business days in the US only!
        """
        us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
        structured_data = pd.date_range(list(train_dates)[-1], periods=self.PREDICTION_DAYS,
                                        freq=us_bd).tolist()
        return structured_data

    def get_prediction(self, model, structured_data):

        forecast_period_dates_three_months = structured_data

        """
        Make prediction
        """
        forecast_three_months = model.predict(self.for_model_prediction)

        """
        inverse transform to rescale back to original numbers
        """
        forecast_copies_three_months = np.repeat(forecast_three_months, self.repeat_df_for_training, axis=-1)
        y_pred_future_three_months = self.scaler.inverse_transform(forecast_copies_three_months)[:, 0]

        """
        Convert timestamp to date
        """
        PREDICTION_DATES = []
        for time_i in forecast_period_dates_three_months:
            PREDICTION_DATES.append(time_i.date())

        """
        Put results into DataFrame
        """
        result1 = pd.DataFrame({'date': np.array(PREDICTION_DATES), 'close': y_pred_future_three_months})
        result2 = np.array(result1.iloc[[0, 9, 59], :])

        prices = {
            'nd_date': result2[0][0],
            'nd_exp_price': result2[0][1],
            'st_date': result2[1][0],
            'st_exp_price': result2[1][1],
            'mt_date': result2[2][0],
            'mt_exp_price': result2[2][1],
        }

        return prices

        # return result, result[:, 0], result[:, 1]
