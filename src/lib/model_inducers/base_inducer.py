from src.lib.clients import FrontendClient
from abc import ABC, abstractmethod


class BaseInducer(ABC):
    def __init__(self, model_save_path):
        super().__init__()
        self.model_save_path = model_save_path
        self.frontend = FrontendClient


    @abstractmethod
    def get_data(self, *args, **kwargs):
        """
            Return
            ------
            A Pandas DataFrame
        """
        pass


    @abstractmethod
    def build_train_test_data(self, *args, **kwargs):
        """
            Return
            ------
            An array of train + test datasets
        """
        pass


    @abstractmethod
    def build_model(self, *args, **kwargs):
        """
            Return
            ------
            A scikit-learn / keras model object
        """
        pass


    @abstractmethod
    def train_model(self, *args, **kwargs):
        """
            Return
            ------
            A dictionary with the following fields:
            {
                'model':                model object after training ,
                'accuracy' OR 'rmse':   metric for evaluation of model,
                'parameters:            params of model
            }
        """
        pass


    @abstractmethod
    def build_prediction_data(self, *args, **kwargs):
        pass


    @abstractmethod
    def get_prediction(self, *args, **kwargs):
        pass
