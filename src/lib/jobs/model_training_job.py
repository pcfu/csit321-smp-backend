from .base_job import BaseJob
from rq import get_current_job


class ModelTrainingJob(BaseJob):
    def __init__(self, config_id, stock_id, date_start, date_end):
        super().__init__()
        self.config_id = config_id
        self.stock_id = stock_id
        self.date_start = date_start
        self.date_end = date_end

        self._check_vars()


    def run(self, *args, **kwargs):
        job = get_current_job()

        try:
            # Update ModelTraining in db to 'training'
            job.meta['status'] = 'started'
            job.save_meta()
            print(f'Performing job - #{job.get_id()}')
            job.meta['status'] = 'done'
            # Update ModelTraining in db to 'done'

        except Exception as err:
            job.meta['status'] = 'error'
            job.meta['message'] = str(err)
            # Update ModelTraining in db to 'error'

        job.save_meta()


    def _check_vars(self):
        if not self._validate_date_format(self.date_start):
            self._raise_date_error('date_start', self.date_start)

        if not self._validate_date_format(self.date_end):
            self._raise_date_error('date_end', self.date_end)


### Object Serialization Example
# serialized = pickle.dumps(job)
# current_app.redis.set('pickled_obj', serialized)
# deserialized = pickle.loads(current_app.redis.get('pickled_obj'))
