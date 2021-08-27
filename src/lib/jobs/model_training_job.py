from .base_job import BaseJob
from rq import get_current_job


REQUIRED_PARAMS = ['model_config', 'stock']


class ModelTrainingJob(BaseJob):
    def run(self, *args, **kwargs):
        job = get_current_job()

        try:
            self.check_params(kwargs, keys=REQUIRED_PARAMS)
            job.meta['status'] = 'started'
            job.save_meta()

            raise RuntimeError('test error')
            job.meta['status'] = 'done'
        except Exception as err:
            job.meta['status'] = 'error'
            job.meta['message'] = str(err)

        job.save_meta()
