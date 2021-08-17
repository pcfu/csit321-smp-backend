import math, yaml, requests
from functools import reduce


class Iex:
    _MODES = ['actual', 'sandbox']
    _METHODS = ['get', 'post']
    _TOKENS = yaml.load(
        open('src/lib/clients/api_keys.yml', 'r'),
        Loader=yaml.Loader
    ).get('iex')
    _CREDIT_LIMIT = 50000
    _PRC_HIST_ATTRS = ['date', 'open', 'high', 'low', 'close', 'volume', 'change']


    def __init__(self, mode='sandbox'):
        self._mode = None
        self._target_domain = None
        self._token = None
        self._sk_token = None
        self.version = 'stable'
        self.set_mode(mode)


    def set_mode(self, mode):
        if mode not in self._MODES:
            raise ValueError(f'Unknown mode "{mode}"')
        self._mode = mode
        self._target_domain = 'cloud.iexapis.com' if mode == 'actual' else 'sandbox.iexapis.com'
        self._token = self._TOKENS[0].get(mode).get('token')
        self._sk_token = self._TOKENS[0].get(mode).get('sk_token')


    def desc(self):
        return {
            'mode': self._mode,
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
        limit = self._CREDIT_LIMIT if self._mode == 'actual' else math.inf
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


    def get_batch_prices(self, symbols, range='2d', prev_day_only=True):
        if type(symbols) == list:
            symbols = reduce(lambda a, b: f'{a},{b}', symbols)
        if prev_day_only:
            range = '2d'

        params ={ 'symbols': symbols, 'types': 'chart', 'range': range }
        res = self._call_api('stock/market/batch', **params)
        if res['status'] == 'error':
            return self._error_response(res['error'])

        if prev_day_only:
            prices = self._previous_day_prices(res['response'].json())
        else:
            prices = self._format_prices(res['response'].json())
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


    def _format_prices(self, raw_data):
        prices = {}

        for symbol, data in raw_data.items():
            prices[symbol] = []
            for day_prices in data['chart']:
                prc_entry = { k: v for k, v in day_prices.items() if k in self._PRC_HIST_ATTRS }
                prc_entry['percent_change'] = day_prices['changePercent']
                prices[symbol].append(prc_entry)

        return prices


    def _previous_day_prices(self, raw_data):
        prices = {}

        for symbol, data in raw_data.items():
            last_prices = data['chart'][-1]
            prices[symbol] = { k: v for k, v in last_prices.items() if k in self._PRC_HIST_ATTRS }
            prices[symbol]['percent_change'] = last_prices['changePercent']

        return prices
