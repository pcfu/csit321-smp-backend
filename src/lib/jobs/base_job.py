from abc import ABC, abstractmethod
from .validation_mixin import ValidationMixin


class BaseJob(ABC, ValidationMixin):
    @abstractmethod
    def run(self, *args, **kwargs):
        pass


    @abstractmethod
    def _check_vars(self):
        pass
