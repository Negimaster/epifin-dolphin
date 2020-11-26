import os
import unittest
import math
from network import RestManager
from portfolio import Portfolio
import urllib3
urllib3.disable_warnings()


class TestPortfolio(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.r = RestManager()

    def setUp(self):
        if os.path.isfile("full.csv"):
            self.port = Portfolio("full.csv")
        else:
            self.port = Portfolio()
        self.p = self.r.getPortfolio(self.port.portfolioid)
        # print(self.p)
        self.p = self.p['values']['2016-06-01']
        self.p = [[e['asset']['asset'], e['asset']['quantity']]
                  for e in self.p]

    def test_portfolio_retrieve_valid(self):
        for asset, qty in self.p:
            self.port.dataframe.at[asset, "quantity"] = qty
        self.port.update_ttvalue()
        self.port.update_nav()
        self.assertTrue(self.port.is_valid())

    def test_portfolio_sharpe(self):
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
