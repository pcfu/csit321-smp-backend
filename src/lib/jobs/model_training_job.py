from .base_job import BaseJob
from rq import get_current_job


class ModelTrainingJob(BaseJob):
    def __init__(self, config_id, stock_id, date_s, date_e):
        super().__init__()
        self.config_id = config_id
        self.stock_id = stock_id
        self.date_s = date_s
        self.date_e = date_e

        self._check_vars()


    def run(self, *args, **kwargs):
        job = get_current_job()

        try:
            prices = self.get_prices()
            self.save_job_status(job, 'prices retrieved')

            # Update ModelTraining in db to 'training'
            self.save_job_status(job, 'training started')

            print(f'Performing job - #{job.get_id()}')
            self.save_job_status(job, 'training completed')
            # Update ModelTraining in db to 'done'

        except Exception as err:
            self.save_job_status(job, 'error', str(err))
            # Update ModelTraining in db to 'error'


    def _check_vars(self):
        if not self._validate_date_format(self.date_s):
            self._raise_date_error('date_s', self.date_s)

        if not self._validate_date_format(self.date_e):
            self._raise_date_error('date_e', self.date_e)


    def get_prices(self):
        res = self.frontend.get_price_histories(self.stock_id, self.date_s, self.date_e)
        if res.get('status') == 'error':
            raise RunetimeError(res.get('raw_response').reason)
        return res.get('data').get('price_histories')


    def save_job_status(self, job, status, message=None):
        job.meta = {}
        job.meta['status'] = status
        if message:
            job.meta['message'] = message
        else:
            job.meta.pop('message', None)
        job.save_meta()


### Object Serialization Example
# serialized = pickle.dumps(job)
# current_app.redis.set('pickled_obj', serialized)
# deserialized = pickle.loads(current_app.redis.get('pickled_obj'))
