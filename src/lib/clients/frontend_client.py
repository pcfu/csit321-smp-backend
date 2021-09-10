import os, requests
from flask import current_app
from config import get_config


def handle_http_errors(fn):
    def wrapper(method, endpoint, **kwargs):
        try:
            return fn(method, endpoint, **kwargs)
        except requests.exceptions.HTTPError as err:
            return {
                'status': 'error',
                'code': err.response.status_code,
                'reason': err.response.reason,
                'raw_response': err.response
            }

    return wrapper


class FrontendClient:
    METHODS = ['get', 'post', 'put']

    @staticmethod
    def get_price_histories(stock_id, start=None, end=None, fields=None):
        endpoint = f'/stocks/{stock_id}/price_histories'
        params = { 'date_start': start, 'date_end': end, 'fields': fields }
        return FrontendClient.call_api('get', endpoint, json=params)


    @staticmethod
    def update_model_training(tid, stage, rmse=None, error_message=None):
        endpoint = f'/admin/model_trainings/{tid}'
        params = { 'stage': stage, 'rmse': rmse, 'error_message': error_message }
        return FrontendClient.call_api('put', endpoint, json=params)


    @staticmethod
    def insert_price_prediction(prediction):
        endpoint = f'/admin/price_predictions'
        params = { 'price_prediction': prediction }
        return FrontendClient.call_api('post', endpoint, json=params)


    @staticmethod
    @handle_http_errors
    def call_api(method, endpoint, **kwargs):
        method = method.lower()
        if method not in FrontendClient.METHODS:
            raise ValueError(f'Unknown method: {method}')
        fn = getattr(requests, method)

        url = FrontendClient.build_url(endpoint)
        res = fn(url, **kwargs)
        res.raise_for_status()
        return { 'status': 'ok', 'data': res.json() }


    @staticmethod
    def build_url(endpoint):
        c_app = current_app
        if not c_app:
            from app import app as c_app

        if endpoint[0] != '/':
            endpoint = '/' + endpoint
        return f'{c_app.frontend_url}{endpoint}'
