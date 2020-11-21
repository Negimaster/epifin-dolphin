import pandas as pd
import numpy as np
import math
from datetime import datetime
from network import RestManager


class Portfolio:

    def __init__(self, dataframe=None, restManager=RestManager(),
                 START_DATE=None, END_DATE=None):
        self.r = restManager
        if not isinstance(self.r, RestManager):
            raise RuntimeError(
                "restManager param should be a valid RestManager instance")
        if START_DATE is None:
            self.START_DATE = datetime(2016, 6, 1).isoformat('T')
        if END_DATE is None:
            self.END_DATE = datetime(2020, 9, 30).isoformat('T')
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
                self.dataframe.loc[i, 'ROIType'] = (resp[str(i)]['13']['type'])

            self.dataframe['ROI'] = pd.to_numeric(
                self.dataframe['ROI'], errors='coerce')
            self.dataframe['annualROI'] = pd.to_numeric(
                self.dataframe['annualROI'], errors='coerce')
            self.dataframe['sharpe'] = pd.to_numeric(
                self.dataframe['sharpe'], errors='coerce')
            self.dataframe['stdDev'] = pd.to_numeric(
                self.dataframe['stdDev'], errors='coerce')

            self.dataframe['quantity'] = 0
            self.dataframe['totalValue'] = 0
            self.dataframe['NAVPercentage'] = 0
        else:
            self.dataframe = dataframe.copy()

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
        for nb, i in enumerate(self.dataframe.index):
            correlationResp = self.r.putRatio(
                [11], b.get_index(), i, self.START_DATE, self.END_DATE, 'yearly')
            for j in self.dataframe.index:
                if correlationResp[str(j)]['11']['type'] == 'double' and self.cov.loc[i, j] is None:
                    self.cov.loc[i, j] = float(
                        correlationResp[str(j)]['11']['value'].replace(',', '.'))
                    self.cov.loc[j, i] = self.cov.loc[i, j]
                else:
                    self.cov.loc[i, j] = float('nan')
            print("{} / {}".format(nb, len(self.dataframe.index)))
        return self.cov

    def get_covariance_unused(self, i, j):
        return self.cov.loc[i, j]

    def get_covariance(self, i, j):
        if self.cov.loc[i, j] is None or math.isnan(self.cov.loc[i, j]):
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
        print((stdDev_i, stdDev_j))
        print(correlation)
        return correlation * stdDev_i * stdDev_j

    # Compute sharpe of portfolio
    def get_variance(self):
        sum = 0
        for i in self.dataframe[self.dataframe['NAVPercentage'] != 0].index:
            for j in self.dataframe[self.dataframe['NAVPercentage'] != 0].index:
                wi = self.dataframe.loc[i, 'NAVPercentage']
                wj = self.dataframe.loc[j, 'NAVPercentage']
                cov = self.get_covariance(i, j)
                print((wi, wj))
                print(cov)
                sum += wi * wj * cov
        return sum

    def get_rendement(self):
        return (self.dataframe['NAVPercentage'] * self.dataframe['annualROI']).sum()

    def get_sharpe(self):
        rendement = self.get_rendement()
        variance = self.get_variance()
        print((rendement, np.sqrt(variance)))
        if variance == 0:
            return 'error'
        return (rendement - 0.0005) / np.sqrt(variance)


if __name__ == "__main__":
    r = RestManager()
    p = Portfolio(restManager=r)
    print(p.get_sharpe())
    print(p.get_dataframe())
    #test = p.get_dataframe().sort_values(by=['sharpe'], ascending=False)
    #print(test[test.assetType == 'PORTFOLIO'])
