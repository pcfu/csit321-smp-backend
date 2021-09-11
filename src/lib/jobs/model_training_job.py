import pickle
import src.lib.model_inducers as inducers
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
            self._notify_training_started()
            inducer = self._get_inducer()

            model = inducer.build_model(self.model_params)
            trng_options = self.model_params.get('training_options')
            datasets = inducer.build_train_test_data(self._get_prices())
            result = inducer.train_model(model, trng_options, datasets)
            self._notify_training_completed(result.get('rmse'))

            inducer.save_model(model, self.training_id)
            serialized = pickle.dumps(inducer)
            self.app.redis.set(self.training_id, serialized)

        except Exception as err:
            self._notify_error_occurred(str(err))


    def _check_vars(self):
        if not self._validate_date_format(self.date_s):
            self._raise_date_error('date_s', self.date_s)

        if not self._validate_date_format(self.date_e):
            self._raise_date_error('date_e', self.date_e)


    def _get_inducer(self):
        inducer_class = getattr(inducers, self.model_params.get('model_class'))
        return inducer_class()


    def _get_prices(self):
        self._save_job_status('retrieving stock prices')
        fields = self.model_params['data_fields']
        params = [ self.stock_id, self.date_s, self.date_e, fields ]

        res = self.frontend.get_price_histories(*params)
        if res.get('status') == 'error':
            raise RuntimeError(res.get('reason'))
        return res.get('price_histories')


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
