from datetime import datetime
import pickle
import src.lib.model_inducers as inducers
from .base_job import BaseJob


class ModelTrainingJob(BaseJob):
    def __init__(self, training_id, stock_id, model_class, model_params):
        super().__init__()
        self.training_id    = training_id
        self.stock_id       = stock_id
        self.model_class    = model_class
        self.model_params   = model_params


    def run(self, *args, **kwargs):
        try:
            self._notify_training_started()
            inducer = self._get_model_inducer()
            date_s = self.model_params.get('start_date')
            date_e = datetime.today().strftime('%Y-%m-%d')

            raw_data = inducer.get_data(self.stock_id, date_s, date_e)
            split_ratio = self.model_params.get('train_test_percent')
            datasets = inducer.build_train_test_data(raw_data, split_ratio)

            model = inducer.build_model(self.model_params)
            result = inducer.train_model(model, self.model_params, datasets)

            ### NAMING SCHEME FOR SAVING MODELS
            ### e.g. KEY = SVM_AAPL_STOCKID_1
            inducer.save_model(result.get('model'))
            serialized = pickle.dumps(inducer)
            self.app.redis.set(self.training_id, serialized)
            del result['model']
            self._notify_training_completed(result)

        except Exception as err:
            self._notify_error_occurred(str(err))


    def _check_vars(self):
        pass


    def _get_model_inducer(self):
        inducer_class = getattr(inducers, self.model_class.upper())
        return inducer_class(self.training_id)


    def _notify_training_started(self):
        self._save_job_status('training started')
        res = self.frontend.update_model_training(self.training_id, 'training')
        if res.get('status') == 'error':
            raise RuntimeError(res.get('reason'))


    # UPDATE THIS METHOD TO INCLUDE rmse/accuracy/parameters
    def _notify_training_completed(self, result):
        self._save_job_status('training completed', message=f'rmse = {rmse}')
        res = self.frontend.update_model_training(self.training_id, 'done', rmse=rmse)
        if res.get('status') == 'error':
            raise RuntimeError(res.get('reason'))


    def _notify_error_occurred(self, e_msg):
        self._save_job_status('error', message=e_msg)
        self.frontend.update_model_training(self.training_id, 'error', error_message=e_msg)
