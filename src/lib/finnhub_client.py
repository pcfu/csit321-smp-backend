import requests, time
from dateutil.relativedelta import relativedelta
from datetime import datetime


class Finnhub:
    _MODES = ['actual', 'sandbox']
    _DATE_FMT = "%Y-%m-%d"
    _API_LIMIT_RESET = 60  # 60 seconds until next API call is definitely unrestricted


    def __init__(self, mode='sandbox'):
        self._mode = None
        self._target_domain = 'finnhub.io/api/v1'
        self._token = None
        self.set_mode(mode)


    def set_mode(self, mode):
        if mode not in self._MODES:
            raise ValueError(f'Unknown mode "{mode}"')

        self._mode = mode
        if mode == 'actual':
            self._token = 'c3icbfaad3ib8lb83050'
        else:
            self._token = 'sandbox_c3icbfaad3ib8lb8305g'


    def desc(self):
        return {
            'mode': self._mode,
            'domain': self._target_domain,
            'token': self._token
        }


    def get_news(self, symbol, from_date=None, to_date=None):
        if from_date is None:
            from_date = self._yesterday()
        if to_date is None:
            to_date = self._today()
        if self._invalid_date_range(from_date, to_date):
            raise ValueError(f'Invalid date range: {from_date} - {to_date}')

        params = { 'symbol': symbol, 'from': from_date, 'to': to_date }
        res = self._call_api('company-news', **params)

        if res['status'] == 'error':
            return res
        else:
            return { 'status': 'ok', 'data': res['response'].json() }


    def get_headlines(self, symbol, start_date=None):
        if start_date is None:
            start_date = datetime.now() - relativedelta(years=1)
        else:
            start_date = datetime.strptime(start_date, self._DATE_FMT)
        end_date = datetime.now()

        headlines = []
        while start_date <= end_date:
            from_date = datetime.strftime(start_date, self._DATE_FMT)
            to_date = datetime.strftime(start_date + relativedelta(days=6), self._DATE_FMT)
            start_date += relativedelta(days=7)

            res = self.get_news(symbol, from_date, to_date)
            headlines += self._format_headlines(res['data'])

        headlines.sort(key = lambda hl: hl.get('datetime'))
        return { 'status': 'ok', 'data': headlines }


    def get_batch_headlines(self, symbols, start_date=None):
        headlines = {}
        for idx, symbol in enumerate(symbols, start=1):
            res = self.get_headlines(symbol, start_date)
            headlines[symbol] = res['data']
            if idx < len(symbols):
                # sleep so that next call is at least 1 minute after current call
                time.sleep(self._API_LIMIT_RESET)

        return { 'status': 'ok', 'data': headlines }


    def _call_api(self, endpoint, **params):
        url = f'https://{self._target_domain}/{endpoint}'
        params['token'] = self._token

        try:
            res = requests.get(url, params=params)
            res.raise_for_status()
            return { 'status': 'ok', 'response': res }
        except requests.exceptions.HTTPError as err:
            return { 'status': 'error', 'error': err.response }


    def _format_headlines(self, raw_data):
        headlines = []

        for data in raw_data:
            dt = datetime.fromtimestamp(data['datetime'])
            entry = {
                'datetime': dt.isoformat(sep=' ', timespec='seconds'),
                'headline': data['headline']
            }
            headlines.append(entry)

        return headlines


    def _today(self):
        return datetime.now().strftime(self._DATE_FMT)


    def _yesterday(self):
        dt = datetime.now() - relativedelta(days=1)
        return dt.strftime(self._DATE_FMT)


    def _invalid_date_range(self, from_date, to_date):
        dt_fr = datetime.strptime(from_date, self._DATE_FMT)
        dt_to = datetime.strptime(to_date, self._DATE_FMT)
        return dt_fr > dt_to
