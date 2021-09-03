import time
from .base_job import BaseJob


class PricePredictionJob(BaseJob):
    def __init__(self, training_id, stock_id, dates):
        super().__init__()
        self.training_id    = training_id
        self.stock_id       = stock_id
        self.date_s         = dates[0]
        self.date_e         = dates[1]

        self._check_vars()


    def run(self, *args, **kwargs):
        try:
            prices = self._get_prices()

            ### PLACEHOLDER CODE ###
            print(f'Simulating price prediction for Stock {self.stock_id} ' +
                  f'using ModelTraining {self.training_id}')
            time.sleep(5)

            ### De-serialize trained model
            serialized = self.app.redis.get(self.training_id)
            model = self.pickle.loads(serialized)
            print("========= MODEL DESCRIPTION ==========")
            print(model.get_desc())
            print("========= MODEL PARAMETERS ===========")
            print(model.get_params())
            print("======================================")
            ### END PLACEHOLDER CODE ###

        except Exception as err:
            self._notify_error_occurred(str(err))


    def _get_prices(self):
        self._save_job_status('retrieving stock prices')
        res = self.frontend.get_price_histories(self.stock_id, self.date_s, self.date_e)
        if res.get('status') == 'error':
            raise RuntimeError(res.get('reason'))
        return res.get('data').get('price_histories')


    def _check_vars(self):
        if not self._validate_date_format(self.date_s):
            self._raise_date_error('date_s', self.date_s)

        if not self._validate_date_format(self.date_e):
            self._raise_date_error('date_e', self.date_e)


    def _notify_error_occurred(self, e_msg):
        self._save_job_status('error', message=e_msg)
        ###
        # send notification to frontend about error
