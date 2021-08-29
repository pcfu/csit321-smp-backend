import os, requests
from flask import current_app
from config import get_config


def handle_http_errors(fn):
    def wrapper(endpoint, **kwargs):
        try:
            return fn(endpoint, **kwargs)
        except requests.exceptions.HTTPError as err:
            return {
                'status': 'error',
                'code': err.response.status_code,
                'raw_response': err.response
            }

    return wrapper


class FrontendClient:
    @staticmethod
    @handle_http_errors
    def get(endpoint, **kwargs):
        url = FrontendClient.build_url(endpoint)
        res = requests.get(url, **kwargs)
        res.raise_for_status()
        return { 'status': 'ok', 'data': res.json() }


    @staticmethod
    def get_price_histories(stock_id, start=None, end=None):
        endpoint = f'/stocks/{stock_id}/price_histories'
        params = { 'date_start': start, 'date_end': end }
        return FrontendClient.get(endpoint, params=params)


    @staticmethod
    def build_url(endpoint):
        c_app = current_app
        if not c_app:
            from app import app as c_app

        if endpoint[0] != '/':
            endpoint = '/' + endpoint
        return f'{c_app.frontend_url}{endpoint}'
