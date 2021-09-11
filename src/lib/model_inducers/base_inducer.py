from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM
from abc import ABC, abstractmethod


class ModelBuilder:
    def __init__(self):
        self.model = None


    def add_model(self, model_class):
        if model_class == 'PricePredictionLSTM':
            self.model = Sequential()
        else:
            raise ValueError(f'unknown model class: {model_class}')


    def add_layer(self, layer):
        layer_type = layer.get('type')
        if layer_type == 'lstm':
            layer_cls = LSTM
        elif layer_type == 'dense':
            layer_cls = Dense
        else:
            raise ValueError(f'unknown layer type: {layer_type}')

        settings = layer.get('settings')
        self.model.add(layer_cls(**settings))


    def compile(self, options):
        self.model.compile(**options)
        return self.model



class BaseInducer(ABC):
    def __init__(self):
        super().__init__()
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model_save_path = None


    @abstractmethod
    def build_train_test_data(self, *args, **kwargs):
        pass


    @abstractmethod
    def train_model(self, *args, **kwargs):
        pass


    def build_model(self, params):
        builder = ModelBuilder()
        builder.add_model(params.get('model_class'))
        for layer in params.get('init_options').get('layers'):
            builder.add_layer(layer)
        return builder.compile(params.get('init_options').get('compile'))


    def save_model(self, model, id):
        self.model_save_path = f'trained_models/{id}'
        model.save(self.model_save_path)


    def load_model(self):
        return load_model(self.model_save_path)
