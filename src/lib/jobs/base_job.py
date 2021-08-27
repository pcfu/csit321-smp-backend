from abc import ABC, abstractmethod


class BaseJob(ABC):
    @abstractmethod
    def run(self, *args, **kwargs):
        pass


    def check_params(self, params, keys=[]):
        for key in keys:
            if key not in params:
                raise ValueError(f'missing keyword argument "{key}"')
