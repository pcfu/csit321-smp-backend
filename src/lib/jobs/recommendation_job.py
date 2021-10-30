import pickle
from datetime import datetime, timedelta
from .base_job import BaseJob


class RecommendationJob(BaseJob):
    def __init__(self, stock_id, model_path, last_date):
        super().__init__()
        self.stock_id   = stock_id
        self.model_path = model_path
        self.last_date  = last_date


    def run(self, *args, **kwargs):
        try:
            serialized = self.app.redis.get(self.model_path)
            inducer = pickle.loads(serialized)
            model = inducer.load_model()

            raw_data = inducer.get_data(self.stock_id, self.last_date, self.last_date)
            structured_data = inducer.build_prediction_data(raw_data)
            recommendation = inducer.get_prediction(model, structured_data)

            params = {
                'stock_id': self.stock_id,
                'prediction_date': self._get_prediction_date(inducer.PREDICTION_DAYS),
                'verdict': recommendation
            }
            self._send_recommendation_to_frontend(params)

        except Exception as err:
            self._save_job_status('error', message=str(err))


    def _get_prediction_date(self, days_delta):
        fmt = '%Y-%m-%d'
        base_dt = datetime.strptime(self.last_date, fmt)
        target_dt = base_dt + timedelta(days = days_delta)
        return datetime.strftime(target_dt, fmt)


    def _send_recommendation_to_frontend(self, params):
        self._save_job_status('recommendation completed')
        res = self.frontend.insert_recommendation(params)
        if res.get('status') == 'error':
            raise RuntimeError(res.get('reason'))
