import time
from datetime import datetime
from random import seed, random
from .base_job import BaseJob


class ModelTrainingJob(BaseJob):
    def __init__(self, training_id, stock_id, model_params, dates):
        super().__init__()
        self.training_id    = training_id
        self.stock_id       = stock_id
        self.model_params   = model_params
        self.date_s         = dates[0]
        self.date_e         = dates[1]

        self._check_vars()


    def run(self, *args, **kwargs):
        try:
            prices = self._get_prices()
            self._notify_training_started()

            ### PLACEHOLDER CODE ###
            print(f'Simulating training for model_training #{self.training_id}')
            time.sleep(5)

            ### Serialize trained model
            # model = LSTM(self.model_params)
            model = DummyModel(self.model_params, self.training_id, \
                               self.stock_id, self.date_s, self.date_e)
            key = self.training_id
            serialized = self.pickle.dumps(model)
            self.app.redis.set(key, serialized)
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


### REMOVE THIS AFTER ACTUAL MODEL PUT IN
class DummyModel:
    def __init__(self, params, tid, sid, date_s, date_e):
        self.params = params
        self.tid = tid
        self.sid = sid
        self.date_s = date_s
        self.date_e = date_e

    def get_params(self):
        return self.params

    def get_desc(self):
        return f'Model was trained for model_training: {self.tid} ' + \
               f'for Stock: {self.sid} ' + \
               f'at prices from {self.date_s} to {self.date_e}'

    def get_prediction(self):
        seed(time.time())
        min_p = 100
        delta = 200

        return {
            'entry_date': datetime.strftime(datetime.now(), "%Y-%m-%d"),
            'nd_min_price': min_p + delta * random(),
            'nd_exp_price': min_p + delta * random(),
            'nd_max_price': min_p + delta * random(),
            'st_min_price': min_p + delta * random(),
            'st_exp_price': min_p + delta * random(),
            'st_max_price': min_p + delta * random(),
            'mt_min_price': min_p + delta * random(),
            'mt_exp_price': min_p + delta * random(),
            'mt_max_price': min_p + delta * random(),
            'lt_min_price': min_p + delta * random(),
            'lt_exp_price': min_p + delta * random(),
            'lt_max_price': min_p + delta * random()
        }
