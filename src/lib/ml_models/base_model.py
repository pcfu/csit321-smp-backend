from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential

from abc import ABC, abstractmethod


class BaseModel(ABC):
    def __init__(self, params):
        super().__init__()
        self.model = None
        self.training_options = params.get('training_options')
        self._build_model(params)


    @abstractmethod
    def buildTrainTestData(self, *args, **kwargs):
        pass


    @abstractmethod
    def train(self, *args, **kwargs):
        pass


    def _build_model(self, params):
        self._add_model(params.get('model'))
        for layer in params.get('init_options').get('layers'):
            self._add_layer(layer)
        self.model.compile(**params.get('init_options').get('compile'))


    def _add_model(self, model_type):
        if model_type == 'sequential':
            self.model = Sequential()
        else:
            raise ValueError(f'unknown model type: {model_type}')


    def _add_layer(self, layer_params):
        layer_type = layer_params.get('type')
        if layer_type == 'lstm':
            layer_cls = LSTM
        elif layer_type == 'dense':
            layer_cls = Dense
        else:
            raise ValueError(f'unknown layer type: {layer_type}')

        settings = layer_params.get('settings')
        self.model.add(layer_cls(**settings))
