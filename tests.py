import os
import unittest
import math
from network import RestManager
from portfolio import Portfolio
import urllib3
urllib3.disable_warnings()


class TestPortfolioRetrieve(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.r = RestManager()

    def setUp(self):
        if os.path.isfile("full.csv"):
            self.port = Portfolio("full.csv", retrieve=True)
        else:
            self.port = Portfolio(retrieve=True)

    def test_portfolio_retrieve_valid(self):
        self.assertTrue(self.port.is_valid(),
                        msg='Invalid retrieved portfolio !')
        self.assertFalse(self.port.has_types())

    def test_portfolio_retrieve_sharpe(self):
        sharpe = self.port.get_sharpe()
        exp_sharpe = self.r.putRatio([12], [self.port.portfolioid], None,
                                     self.port.START_DATE, self.port.END_DATE, None)
        exp_sharpe = float(
            exp_sharpe[str(self.port.portfolioid)]['12']['value'].replace(',', '.'))
        diff = math.fabs(exp_sharpe - sharpe)
        self.assertLessEqual(
            diff, 1e-8, msg=f'exp: {exp_sharpe}, got: {sharpe}')


if __name__ == '__main__':
    unittest.main()
