from src.lib.jobs.ti_calculator import TiCalculator
from .base_job import BaseJob


class TisUpdateJob(BaseJob):
    def __init__(self, stock_id, prices, n_last_data):
        super().__init__()
        self.stock_id = stock_id
        self.prices = prices
        self.n_last_data = n_last_data


    def run(self):
        try:
            tis = TiCalculator.calc_tis(self.prices)[-self.n_last_data:]
            self.frontend.insert_technical_indicators(self.stock_id, tis)
        except Exception as err:
            print(err)


    def _check_vars(self):
        # nothing to check
        pass
