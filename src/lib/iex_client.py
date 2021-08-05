import math, yaml, requests
from functools import reduce


SAMPLE_ENTITIES = {
    'aig'       : 'AIG',
    'amazon'    : 'AMZN',
    'apple'     : 'AAPL',
    'facebook'  : 'FB',
    'google'    : 'GOOG',
    'jpmorgan'  : 'JPM',
    'mcdonalds' : 'MCD',
    'tesla'     : 'TSLA',
    'twitter'   : 'TWTR',
    'walmart'   : 'WMT'
}


class Iex:
    _ENVS = ['production', 'development', 'test']
    _METHODS = ['get', 'post']
    _TOKENS = yaml.load(open('src/lib/api_keys.yml', 'r'), Loader=yaml.Loader)
    _CREDIT_LIMIT = 50000


    def __init__(self, env='development'):
        self._env = env
        self._target_domain = None
        self._token = None
        self._sk_token = None
        self.version = 'stable'
        self.set_env(env)


    def set_env(self, env):
        if env not in self._ENVS:
            raise ValueError(f'Unknown env "{env}"')

        self._env = env
        if env == 'production':
            self._target_domain = 'cloud.iexapis.com'
            self._token = self._TOKENS[0]['cloud']['token']
            self._sk_token = self._TOKENS[0]['cloud']['sk_token']
        else:
            self._target_domain = 'sandbox.iexapis.com'
            self._token = self._TOKENS[0]['sandbox']['token']
            self._sk_token = self._TOKENS[0]['sandbox']['sk_token']


    def desc(self):
        return {
            'env': self._env,
            'domain': self._target_domain,
            'token': self._token,
            'sk_token': self._sk_token,
            'api_version': self.version
        }


    def credits(self):
        res = self._call_api('account/usage/credits', use_sk_token=True)
        if res['status'] == 'error':
            return self._error_response(res['error'])

        data = res['response'].json()
        limit = self._CREDIT_LIMIT if self._env == 'production' else math.inf
        credits = {
            'limit': limit,
            'usage': data['monthlyUsage'],
            'remaining': limit - data['monthlyUsage']
        }

        return { 'status': 'ok', 'data': credits }


    def get_stocks(self, exchange=None):
        res = self._call_api('ref-data/symbols')
        if res['status'] == 'error':
            return self._error_response(res['error'])

        data = res['response'].json()
        if exchange is not None:
            data = list(filter(lambda stock: stock['exchange'] == exchange, data))

        data = list(map(
            lambda stock: {
                'symbol': stock['symbol'],
                'name': stock['name'],
                'exchange': stock['exchange'],
                'stock_type': stock['type'],
            },
            data
        ))

        return { 'status': 'ok' , 'data': data }


    def get_prices(self, symbol, range='1y'):
        res =  self._call_api(f'stock/{symbol.lower()}/chart/{range}')
        if res['status'] == 'error':
            return self._error_response(res['error'])

        data = res['response'].json()
        return { 'status': 'ok' , 'data': data }


    def get_batch_prices(self, symbols=None, range='2d'):
        if symbols is None:
            symbols = list(SAMPLE_ENTITIES.values())
        symbols = reduce(lambda a, b: f'{a},{b}', symbols)

        params ={ 'symbols': symbols, 'types': 'chart', 'range': range }
        res = self._call_api('stock/market/batch', **params)
        if res['status'] == 'error':
            return self._error_response(res['error'])

        prices = self._previous_day_prices(res['response'].json())
        return { 'status': 'ok' , 'data': prices }


    def _call_api(self, endpoint, method='get', use_sk_token=False, **params):
        if method not in self._METHODS:
            raise ValueError('Invalid method')

        url = f'https://{self._target_domain}/{self.version}/{endpoint}'
        params['token'] = self._sk_token if use_sk_token else self._token

        try:
            if method == 'get':
                res = requests.get(url, params=params)
            else:
                res = requests.post(url, params=params)
            res.raise_for_status()
            return { 'status': 'ok', 'response': res }

        except requests.exceptions.HTTPError as err:
            return { 'status': 'error', 'error': err.response }


    def _error_response(self, err):
        return {
            'status': 'error',
            'message': f'{err.status_code} - {err.reason}'
        }


    def _previous_day_prices(self, raw_data):
        keys = ['date', 'open', 'high', 'low', 'close', 'volume', 'change']
        prices = {}

        for symbol, data in raw_data.items():
            last_prices = data['chart'][-1]
            prices[symbol] = {k: v for k, v in last_prices.items() if k in keys}
            prices[symbol]['change_percent'] = last_prices['changePercent']

        return prices
