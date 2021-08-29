import time
from .base_job import BaseJob


class ModelTrainingJob(BaseJob):
    def __init__(self, training_id, config_id, stock_id, date_s, date_e):
        super().__init__()
        self.training_id = training_id
        self.config_id = config_id
        self.stock_id = stock_id
        self.date_s = date_s
        self.date_e = date_e

        self._check_vars()


    def run(self, *args, **kwargs):
        try:
            prices = self._get_prices()
            self._notify_training_started()

            ### PLACEHOLDER CODE ###
            print(f'Simulating training for model_training #{self.training_id}')
            time.sleep(10)

            ### Serialize (De-serialize in prediction job) trained model
            # key = self.training_id
            # serialized = self.pickle.dumps( <MODEL>)
            # self.app.redis.set(key, serialized)
            # deserialized = self.pickle.loads( self.app.redis.get(key) )
            ### END PLACEHOLDER CODE ###

            self._notify_training_completed(12.345)   # with placeholder rmse

        except Exception as err:
            self._notify_error_occurred(str(err))


    def _check_vars(self):
        if not self._validate_date_format(self.date_s):
            self._raise_date_error('date_s', self.date_s)

        if not self._validate_date_format(self.date_e):
            self._raise_date_error('date_e', self.date_e)


    def _get_prices(self):
        self._save_job_status('retrieving stock prices')
        res = self.frontend.get_price_histories(self.stock_id, self.date_s, self.date_e)
        if res.get('status') == 'error':
            raise RuntimeError(res.get('reason'))
        return res.get('data').get('price_histories')


    def _notify_training_started(self):
        self._save_job_status('training started')
        res = self.frontend.update_model_training(self.training_id, 'training')
        if res.get('status') == 'error':
            raise RuntimeError(res.get('reason'))


    def _notify_training_completed(self, rmse):
        self._save_job_status('training completed', message=f'rmse = {rmse}')
        res = self.frontend.update_model_training(self.training_id, 'done', rmse=rmse)
        if res.get('status') == 'error':
            raise RuntimeError(res.get('reason'))


    def _notify_error_occurred(self, e_msg):
        self._save_job_status('error', message=e_msg)
        self.frontend.update_model_training(self.training_id, 'error', error_message=e_msg)
