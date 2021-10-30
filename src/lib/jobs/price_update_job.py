from .base_job import BaseJob
from src.lib.clients import Iex


class PriceUpdateJob(BaseJob):
    def __init__(self, symbols, days):
        super().__init__()
        self.symbols = symbols
        self.days = days


    def run(self):
        try:
            cl = Iex('actual')
            res = cl.get_batch_prices(self.symbols, self.days)
            data = res.get('data')
            params = [{'symbol': k, 'prices': v[::-1]} for k, v in data.items()]
            self.frontend.insert_price_histories(params)

        except Exception as err:
            print(err)
