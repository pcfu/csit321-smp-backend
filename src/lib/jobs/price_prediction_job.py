import pickle
from datetime import datetime, timedelta
from .base_job import BaseJob


class PricePredictionJob(BaseJob):
    RAW_DATA_RANGE = 365    # 1 year of raw data
    DATE_FORMAT    = "%Y-%m-%d"


    def __init__(self, stock_id, model_path, last_date):
        super().__init__()
        self.stock_id    = stock_id
        self.model_path = model_path
        self.last_date  = last_date


    def run(self, *args, **kwargs):
        try:
            serialized = self.app.redis.get(self.model_path)
            inducer = pickle.loads(serialized)
            model = inducer.load_model()

            start_date = self._get_start_date()
            raw_data = inducer.get_data(self.stock_id, start_date, self.last_date)
            structured_data = inducer.build_prediction_data(*raw_data)
            prices = inducer.get_prediction(model, *structured_data)

            prediction_params = self._build_prediction_params(prices)
            self._send_prediction_to_frontend(prediction_params)

        except Exception as err:
            self._notify_error_occurred(str(err))


    def _get_start_date(self):
        date_e = datetime.strptime(self.last_date, PricePredictionJob.DATE_FORMAT)
        date_s = date_e - timedelta(days = PricePredictionJob.RAW_DATA_RANGE - 1)
        return datetime.strftime(date_s, PricePredictionJob.DATE_FORMAT)


    def _build_prediction_params(self, prices):
        prices['stock_id'] = self.stock_id
        prices['reference_date'] = self.last_date
        return prices


    def _send_prediction_to_frontend(self, prediction):
        self._save_job_status('prediction completed')
        res = self.frontend.insert_price_prediction(prediction)
        if res.get('status') == 'error':
            raise RuntimeError(res.get('reason'))


    def _notify_error_occurred(self, e_msg):
        self._save_job_status('error', message=e_msg)
