import pandas as pd
import numpy as np
import math
import os
import warnings
from datetime import datetime
from network import RestManager


class Portfolio:

    def __init__(self, path=None, dataframe=None, restManager=RestManager(),
                 START_DATE=None, END_DATE=None):
        self.r = restManager
        if not isinstance(self.r, RestManager):
            raise RuntimeError(
                "restManager param should be a valid RestManager instance")
        if START_DATE is None:
            self.START_DATE = datetime(2016, 6, 1).isoformat('T')
        if END_DATE is None:
            self.END_DATE = datetime(2020, 9, 30).isoformat('T')
        if path is not None and os.path.isfile(path):
            self.load_portfolio(path)
        else:
            if dataframe is None:
                pars = self.r.getAssetList(self.START_DATE)
                company_id_array = []
                company_name_array = []
                asset_currency_array = []
                asset_type_array = []
                asset_value_array = []

                for asset in pars:
                    company_id_array.append(
                        int(asset['ASSET_DATABASE_ID']['value']))
                    company_name_array.append(asset['LABEL']['value'])
                    asset_currency_array.append(asset['CURRENCY']['value'])
                    asset_type_array.append(asset['TYPE']['value'])
                    asset_value_array.append(asset['LAST_CLOSE_VALUE_IN_CURR']['value'].split()[0].replace(
                        ',', '.')) if 'LAST_CLOSE_VALUE_IN_CURR' in asset else asset_value_array.append('error')

                self.dataframe = pd.DataFrame(index=company_id_array)
                self.dataframe['assetName'] = company_name_array
                self.dataframe['assetCurrency'] = asset_currency_array
                self.dataframe['assetType'] = asset_type_array
                self.dataframe['assetValue'] = asset_value_array
                self.dataframe['assetValue'] = pd.to_numeric(
                    self.dataframe['assetValue'], errors='coerce')

                resp = self.r.putRatio([13, 12, 10, 9], company_id_array,
                                       None, self.START_DATE, self.END_DATE, 'yearly')

                self.dataframe['ROI'] = None
                self.dataframe['annualROI'] = None
                self.dataframe['sharpe'] = None
                self.dataframe['stdDev'] = None

                for i in company_id_array:
                    if i == 2201:
                        print((resp[str(i)]['12']['value']).replace(',', '.'))
                    self.dataframe.loc[i, 'ROI'] = (
                        resp[str(i)]['13']['value']).replace(',', '.')
                    self.dataframe.loc[i, 'annualROI'] = (
                        resp[str(i)]['9']['value']).replace(',', '.')
                    self.dataframe.loc[i, 'sharpe'] = (
                        resp[str(i)]['12']['value']).replace(',', '.')
                    self.dataframe.loc[i, 'stdDev'] = (
                        resp[str(i)]['10']['value']).replace(',', '.')
                    self.dataframe.loc[i, 'ROIType'] = (
                        resp[str(i)]['13']['type'])

                self.dataframe['ROI'] = pd.to_numeric(
                    self.dataframe['ROI'], errors='coerce')
                self.dataframe['annualROI'] = pd.to_numeric(
                    self.dataframe['annualROI'], errors='coerce')
                self.dataframe['sharpe'] = pd.to_numeric(
                    self.dataframe['sharpe'], errors='coerce')
                self.dataframe['stdDev'] = pd.to_numeric(
                    self.dataframe['stdDev'], errors='coerce')

                self.dataframe['quantity'] = 0
                self.dataframe['totalValue'] = 0.0
                self.dataframe['NAVPercentage'] = 0.0
            else:
                self.dataframe = dataframe.copy()
        self.dataframe = self.dataframe.astype(
            {'totalValue': 'float64', 'NAVPercentage': 'float64'})

        self.cov = pd.DataFrame(index=self.dataframe.index)
        for i in self.dataframe.index:
            self.cov[i] = None

    def get_dataframe(self):
        return self.dataframe

    def get_index(self):
        return self.dataframe.index.tolist()

    def is_fund(self, id):
        return 'FUND' in self.dataframe.loc[id, 'assetType']

    def get_nb_asset(self):
        return len(self.dataframe[self.dataframe['quantity'] != 0])

    def update_nav(self):
        self.dataframe['NAVPercentage'] = self.dataframe['totalValue'] / \
            self.dataframe['totalValue'].sum()

    def get_asset(self, id, n):
        quantity = self.dataframe.loc[id, 'quantity']
        self.dataframe.loc[id, 'quantity'] += n if n + \
            quantity >= 0 or n > 0 else 0
        self.dataframe.loc[id, 'totalValue'] = self.dataframe.loc[id,
                                                                  'assetValue'] * self.dataframe.loc[id, 'quantity']
        self.update_nav()
        return self.dataframe

    def print_cov(self):
        return self.cov

    # Fills in correlation matrix but takes 3 minutes
    def init_correlation(self):
        if os.path.isfile("cov.npy"):
            return self.load_cov()
        for nbi, i in enumerate(self.dataframe.index):
            correlationResp = self.r.putRatio(
                [11], self.get_index(), i, self.START_DATE, self.END_DATE, 'yearly')
            l = []  # np.array(len(correlationResp))
            for j in self.dataframe.index:
                j = str(j)
                if correlationResp[j]['11']['type'] == 'double':
                    l.append(float(
                        correlationResp[j]['11']['value'].replace(',', '.')))
                else:
                    l.append(float("nan"))
            self.cov.loc[i] = l
            self.cov[i] = l
            print(
                f"Loading correlations: {nbi} / {len(self.dataframe.index)}", end="\r")
        # print(self.cov)
        toremove = self.cov.isnull().all(axis=1)
        toremove = [i for i in toremove.index if toremove.loc[i]]
        self.dataframe.drop(toremove, inplace=True)
        self.cov.dropna(axis=0, how="all", inplace=True)
        self.cov.dropna(axis=1, how="all", inplace=True)
        # print(self.cov)
        # print(self.dataframe)
        assert(not self.cov.isnull().values.any())
        if not os.path.isfile("cov.npy"):
            self.dump_cov()
        return self.cov

    def get_covariance_unused(self, i, j):
        return self.cov.loc[i, j]

    def get_covariance(self, i, j):
        if self.cov.loc[i, j] is None or math.isnan(self.cov.loc[i, j]):
            raise RuntimeError(f"Should be init at : {i},{j}")
            correlationResp = self.r.putRatio(
                [11], [i], j, self.START_DATE, self.END_DATE, "yearly")
            if correlationResp[str(i)]['11']['type'] == 'double':
                correlation = float(
                    (correlationResp[str(i)]['11']['value']).replace(',', '.'))
                self.cov.loc[i, j] = correlation
                self.cov.loc[j, i] = correlation
            else:
                correlation = float('NaN')
        else:
            correlation = self.cov.loc[i, j]
        stdDev_i = self.dataframe.loc[i, 'stdDev']
        stdDev_j = self.dataframe.loc[j, 'stdDev']
        # print((stdDev_i, stdDev_j))
        # print(correlation)
        return correlation * stdDev_i * stdDev_j

    # Compute sharpe of portfolio
    def get_variance(self):
        sum = 0
        for i in self.dataframe[self.dataframe['NAVPercentage'] != 0].index:
            for j in self.dataframe[self.dataframe['NAVPercentage'] != 0].index:
                wi = self.dataframe.loc[i, 'NAVPercentage']
                wj = self.dataframe.loc[j, 'NAVPercentage']
                cov = self.get_covariance(i, j)
                # print((wi, wj))
                # print(cov)
                sum += wi * wj * cov
        if sum == 0:
            warnings.warn("Invalid variance is 0 !")
            return float("inf")
        return sum

    def get_rendement(self):
        return (self.dataframe['NAVPercentage'] * self.dataframe['annualROI']).sum()

    def get_sharpe(self):
        rendement = self.get_rendement()
        variance = self.get_variance()
        # print((rendement, np.sqrt(variance)))
        if variance == 0:
            raise RuntimeError('Invalid Variance cannot be zero !')
        r = (rendement - 0.0005) / np.sqrt(variance)
        if math.isnan(r):
            return float("-inf")
        return r

    def __len__(self):
        return self.dataframe.shape[0]

    def load_cov(self, file="cov.npy"):
        self.cov = np.load(file)
        self.cov = pd.DataFrame(self.cov, index=self.dataframe.index)
        self.cov = self.cov.rename(
            columns={n: newname for n, newname in enumerate(self.dataframe.index)})
        assert(not self.cov.isnull().values.any())
        return self.cov

    def dump_cov(self, file="cov.npy"):
        # assert(self.cov.iat[0, 0]
        #       is not None and not math.isnan(self.cov.at[0, 0]))
        np.save(file, np.array(self.cov))

    def load_portfolio(self, file="full.csv"):
        self.dataframe = pd.read_csv("full.csv", index_col=0)
        self.dataframe = self.dataframe.astype(
            {"totalValue": "float64", "NAVPercentage": "float64"})

    def dump_portfolio(self, file="full.csv"):
        self.dataframe.to_csv()

    def is_valid(self):
        nb_different_assets = (self.dataframe['NAVPercentage'] != 0).sum()
        valid_nb_different_assets = 15 <= nb_different_assets and nb_different_assets <= 40

        print(f'valid_nb_different_assets: {valid_nb_different_assets}')

        stock_navs = self.dataframe[(self.dataframe['NAVPercentage'] != 0.0) & (
            self.dataframe['assetType'] == 'STOCK')]['NAVPercentage']
        at_least_half_actions = stock_navs.sum() >= 0.5

        print(f'at_least_half_actions: {at_least_half_actions}')

        non_zero_navs = self.dataframe[self.dataframe['NAVPercentage']
                                       != 0]['NAVPercentage']
        valid_navs = ((non_zero_navs >= 0.01) & (non_zero_navs <= 0.10)).all()

        print(f'valid_navs: {valid_navs}')
        return valid_nb_different_assets and at_least_half_actions and valid_navs


if __name__ == "__main__":
    r = RestManager()
    p = Portfolio(restManager=r)
    print(p.get_sharpe())
    print(p.get_dataframe())
    print(p.is_valid())
    # test = p.get_dataframe().sort_values(by=['sharpe'], ascending=False)
    # print(test[test.assetType == 'PORTFOLIO'])
