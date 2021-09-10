import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from .base_model import BaseModel


class PricePredictionLSTM(BaseModel):
    TRAIN_TEST_RATIO = 0.65
    TOTAL_TIMESTEPS = 100


    def __init__(self, params):
        super().__init__(params)
        self.scaler = MinMaxScaler(feature_range=(0, 1))


    def buildTrainTestData(self, data):
        """
            Returns an array containing four np arrays with scaled data.
            e.g. [ x_train, y_train, x_test, y_test ]

            Parameters
            ----------
            data: list
                An array of json objects of form { close: price }
        """

        scaled_data = self.scaler.fit_transform(pd.DataFrame(data))
        train_size = int(len(scaled_data) * self.TRAIN_TEST_RATIO)
        train_set, test_set = scaled_data[:train_size], scaled_data[train_size:]

        return [
            *self.build_features_and_labels(train_set),
            *self.build_features_and_labels(test_set)
        ]


    def train(self, x_train, y_train, x_test, y_test, verbose=0):
        self.model.fit(
            x_train,
            y_train,
            verbose=verbose,
            **self.training_options
        )

        y_predict = self.scaler.inverse_transform(self.model.predict(x_test))
        return { 'rmse': np.sqrt(mean_squared_error(y_test, y_predict)) }


    def build_features_and_labels(self, data):
        features, labels = [], []
        for i in range(len(data) - self.TOTAL_TIMESTEPS - 1):
            feature_set = data[i:(i + self.TOTAL_TIMESTEPS), 0]
            features.append(feature_set)

            label = data[i + self.TOTAL_TIMESTEPS, 0]
            labels.append(label)

        features_shape = (len(features), self.TOTAL_TIMESTEPS, 1)
        return [ np.array(features).reshape(features_shape), np.array(labels) ]



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
