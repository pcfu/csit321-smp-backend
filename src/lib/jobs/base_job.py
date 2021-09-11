from flask import current_app
from abc import ABC, abstractmethod
from rq import get_current_job
from src.lib.clients import FrontendClient
from .validation_mixin import ValidationMixin


class BaseJob(ABC, ValidationMixin):
    @property
    def frontend(self):
        return FrontendClient


    @property
    def app(self):
        return self._get_app()


    @abstractmethod
    def run(self, *args, **kwargs):
        pass


    @abstractmethod
    def _check_vars(self):
        pass


    def _get_app(self):
        c_app = current_app
        if not c_app:
            from app import app as c_app
        return c_app


    def _save_job_status(self, status, message=None):
        job = get_current_job()
        if not job: return

        job.meta = {}
        job.meta['status'] = status
        if message:
            job.meta['message'] = message
        else:
            job.meta.pop('message', None)
        job.save_meta()
