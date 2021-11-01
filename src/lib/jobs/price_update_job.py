from .base_job import BaseJob
from src.lib.clients import Iex


class PriceUpdateJob(BaseJob):
    def __init__(self, symbol, days):
        super().__init__()
        self.symbol = symbol
        self.days = days


    def run(self):
        try:
            cl = Iex('actual', self.symbol)
            res = cl.get_prices(self.symbol, self.days)
            prices = res.get('data')
            params = { 'symbol': self.symbol, 'prices': prices }
            self.frontend.insert_price_histories(params)

        except Exception as err:
            print(err)
