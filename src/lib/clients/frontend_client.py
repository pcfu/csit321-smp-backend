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
    def get_sister_prices(stock_id, start=None, end=None):
        endpoint = f'/stocks/{stock_id}/price_histories/sister_prices'
        params = { 'date_start': start, 'date_end': end }
        return FrontendClient.call_api('get', endpoint, json=params)


    @staticmethod
    def insert_price_histories(data):
        endpoint = 'stocks/price_histories/batch_create'
        params = { 'price_histories': data }
        return FrontendClient.call_api('post', endpoint, json=params)


    @staticmethod
    def get_technical_indicators(stock_id, start=None, end=None):
        endpoint = f'/stocks/{stock_id}/technical_indicators'
        params = { 'date_start': start, 'date_end': end }
        return FrontendClient.call_api('get', endpoint, json=params)


    @staticmethod
    def insert_technical_indicators(stock_id, data):
        endpoint = f'stocks/{stock_id}/technical_indicators/batch_create'
        params = { 'technical_indicators': data }
        return FrontendClient.call_api('post', endpoint, json=params)


    @staticmethod
    def get_model_config(config_id):
        endpoint = f'/admin/model_configs/{config_id}'
        return FrontendClient.call_api('get', endpoint)


    @staticmethod
    def update_model_training(tid, stage, rmse=None, accuracy=None,
                              parameters=None, error_message=None):
        endpoint = f'/admin/model_trainings/{tid}'
        params = {
            'model_training': {
                'stage': stage,
                'rmse': rmse,
                'accuracy': accuracy,
                'parameters': parameters,
                'error_message': error_message
            }
        }
        return FrontendClient.call_api('put', endpoint, json=params)


    @staticmethod
    def insert_price_prediction(prediction):
        endpoint = '/admin/price_predictions'
        params = { 'price_prediction': prediction }
        return FrontendClient.call_api('post', endpoint, json=params)


    @staticmethod
    def insert_recommendation(recommendation):
        endpoint = '/admin/recommendations'
        params = { 'recommendation': recommendation }
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
        return res.json()


    @staticmethod
    def build_url(endpoint):
        c_app = current_app
        if not c_app:
            from app import app as c_app

        if endpoint[0] != '/':
            endpoint = '/' + endpoint
        return f'{c_app.frontend_url}{endpoint}'
