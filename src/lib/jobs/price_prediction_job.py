import pickle
from datetime import datetime
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
            serialized = self.app.redis.get(self.training_id)
            inducer = pickle.loads(serialized)
            model = inducer.load_model()

            raw_data = inducer.get_data(self.stock_id, self.date_s, self.date_e)
            structured_data = inducer.build_prediction_data(raw_data)
            prices = inducer.get_prediction(model, structured_data)
            prediction_params = self._build_prediction_params(prices)
            self._send_prediction_to_frontend(prediction_params)

        except Exception as err:
            self._notify_error_occurred(str(err))


    def _check_vars(self):
        if not self._validate_date_format(self.date_s):
            self._raise_date_error('date_s', self.date_s)

        if not self._validate_date_format(self.date_e):
            self._raise_date_error('date_e', self.date_e)


    def _build_prediction_params(self, prices):
        return {
            'stock_id': self.stock_id,
            'entry_date': datetime.strftime(datetime.now(), "%Y-%m-%d"),
            'nd_max_price': prices[0],
            'nd_exp_price': prices[0],
            'nd_min_price': prices[0],
            'st_max_price': prices[13],
            'st_exp_price': prices[13],
            'st_min_price': prices[13],
            'mt_max_price': prices[13],
            'mt_exp_price': prices[13],
            'mt_min_price': prices[13],
            'lt_max_price': prices[13],
            'lt_exp_price': prices[13],
            'lt_min_price': prices[13]
        }

    def _send_prediction_to_frontend(self, prediction):
        self._save_job_status('prediction completed')
        res = self.frontend.insert_price_prediction(prediction)
        if res.get('status') == 'error':
            raise RuntimeError(res.get('reason'))


    def _notify_error_occurred(self, e_msg):
        self._save_job_status('error', message=e_msg)
        ###
        # send notification to frontend about error
