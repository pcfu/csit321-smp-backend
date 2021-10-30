import pickle
from datetime import datetime, timedelta
from .base_job import BaseJob


class PricePredictionJob(BaseJob):
    def __init__(self, training_id, stock_id):
        super().__init__()
        self.training_id = training_id
        self.stock_id    = stock_id


    def run(self, *args, **kwargs):
        try:
            serialized = self.app.redis.get(self.training_id)
            inducer = pickle.loads(serialized)
            model = inducer.load_model()

            date_s, date_e = self._get_data_range(inducer.PREDICTION_DAYS)
            raw_data = inducer.get_data(self.stock_id, date_s, date_e)
            structured_data = inducer.build_prediction_data(raw_data)
            prices = inducer.get_prediction(model, structured_data)
            prediction_params = self._build_prediction_params(prices)
            self._send_prediction_to_frontend(prediction_params)

        except Exception as err:
            self._notify_error_occurred(str(err))


    def _get_data_range(self, date_delta):
        fmt = "%Y-%m-%d"
        date_end = datetime.today()
        date_start = date_end - timedelta(days = date_delta - 1)
        return [
            datetime.strftime(date_start, fmt),
            datetime.strftime(date_end, fmt)
        ]


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
