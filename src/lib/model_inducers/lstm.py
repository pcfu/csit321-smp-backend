import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from .base_inducer import BaseInducer


class PricePredictionLSTM(BaseInducer):
    TRAIN_TEST_RATIO = 0.65
    TOTAL_TIMESTEPS = 100
    PREDICTION_DAYS = 14


    def __init__(self, training_id, model_data_fields):
        """
            Parameters
            ----------
            training_id: int | str
                Training id of corresponding ModelTraining in Frontend

            model_data_fields: list
                A list of strings indicating fields to retrieve for get_data
        """

        model_save_path = f'trained_models/{training_id}'
        super().__init__(model_save_path, model_data_fields)


    def train_model(self, model, options, datasets, verbose=0):
        """
            Trains the specified model using given options and data.
            Returns a dict containing training results.
            e.g. { 'rmse': float }

            Parameters
            ----------
            model: object
                tf.keras Sequential model

            options: dict
                Model training options

            datasets: list
                Contains four np arrays of the following shapes:
                    x_train: shape = ( train_size, TOTAL_TIMESTEPS, 1 )
                    y_train: shape = ( train_size, )
                    x_test:  shape = ( test_size, TOTAL_TIMESTEPS, 1 )
                    y_test:  shape = ( test_size, )
        """

        x_train, y_train, x_test, y_test = datasets
        model.fit(x_train, y_train, verbose=verbose, **options)

        y_predict = self.scaler.inverse_transform(model.predict(x_test))
        return { 'rmse': np.sqrt(mean_squared_error(y_test, y_predict)) }


    def get_prediction(self, model, data):
        results = []
        elements = data.reshape(-1).tolist()

        for _ in range(self.PREDICTION_DAYS):
            if len(elements) > self.TOTAL_TIMESTEPS:
                elements = elements[1:]
                data_size = len(elements)
                data = np.array(elements).reshape(1, data_size, 1)
            else:
                data_size = len(data.reshape(-1))
                data.shape = (1, data_size, 1)

            res = model.predict(data, verbose=0)
            elements.extend(res[0].tolist())
            results.extend(res.tolist())

        return np.reshape(results, (-1))


    def get_data(self, stock_id, date_start, date_end):
        params = [ stock_id, date_start, date_end, self.model_data_fields ]
        res = self.frontend.get_price_histories(*params)
        if res.get('status') == 'error':
            raise RuntimeError(res.get('reason'))
        return res.get('price_histories')


    def build_train_test_data(self, data):
        """
            Returns a list containing four np arrays with scaled data.
            e.g. [
                x_train: shape = ( train_size, TOTAL_TIMESTEPS, 1 )
                y_train: shape = ( train_size, )
                x_test:  shape = ( test_size, TOTAL_TIMESTEPS, 1 )
                y_test:  shape = ( test_size, )
            ]

            Parameters
            ----------
            data: list
                An array of json objects of form { 'close': float }
        """

        scaled_data = self.scaler.fit_transform(pd.DataFrame(data))
        train_size = int(len(scaled_data) * self.TRAIN_TEST_RATIO)
        train_set, test_set = scaled_data[:train_size], scaled_data[train_size:]

        return [
            *self.build_features_and_labels(train_set),
            *self.build_features_and_labels(test_set)
        ]


    def build_prediction_data(self, data):
        """
            Parameters
            ----------
            data: list
                An array of json objects of form { 'close': float }
        """

        df = pd.DataFrame(data)
        return self.scaler.fit_transform(df).reshape(1, -1)


    def build_features_and_labels(self, data):
        features, labels = [], []
        for i in range(len(data) - self.TOTAL_TIMESTEPS - 1):
            feature_set = data[i:(i + self.TOTAL_TIMESTEPS), 0]
            features.append(feature_set)

            label = data[i + self.TOTAL_TIMESTEPS, 0]
            labels.append(label)

        features_shape = (len(features), self.TOTAL_TIMESTEPS, 1)
        return [ np.array(features).reshape(features_shape), np.array(labels) ]
