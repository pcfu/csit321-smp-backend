import pickle
from flask import current_app
from abc import ABC, abstractmethod
from src.lib.clients import FrontendClient
from .validation_mixin import ValidationMixin


class BaseJob(ABC, ValidationMixin):
    @abstractmethod
    def __init__(self):
        super().__init__()
        self.frontend = FrontendClient


    @abstractmethod
    def run(self, *args, **kwargs):
        pass


    @abstractmethod
    def _check_vars(self):
        pass


    def get_app(self):
        c_app = current_app
        if not c_app:
            from app import app as c_app
        return c_app


    def get_pickle(self):
        return pickle
