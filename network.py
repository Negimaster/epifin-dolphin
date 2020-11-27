from requests import Session
from requests.compat import urljoin
import requests
from base64 import b64encode
import urllib3
from datetime import datetime
import pandas as pd
import warnings

# Environment
from dotenv import load_dotenv
import os
from os import getenv

if not os.path.isfile('.env'):
    username = ''
    password = ''
    warnings.warn('.env file not present ! creating...')
    while not username or not password:
        username = input('DOLPHIN_USERNAME')
        password = input('DOLPHIN_PASSWORD')
    with open('.env', 'w') as f:
        f.write(f'DOLPHIN_USERNAME={username}\n')
        f.write(f'DOLPHIN_PASSWORD={password}')
# Load the .env environment variables
load_dotenv()
urllib3.disable_warnings()


class NetworkManager(object):
    '''
    Low level network manager class; python requests module wrapper
    '''

    def __init__(self):
        # Username and Password
        self.__USERNAME = getenv('DOLPHIN_USERNAME')
        self.__PASSWORD = getenv('DOLPHIN_PASSWORD')
        if not self.__USERNAME or not self.__PASSWORD:
            raise RuntimeError(
                'Variable DOLPHIN_USERNAME or DOLPHIN_PASSWORD is not set')
        self.__USERNAME = bytes(self.__USERNAME, 'utf-8')
        self.__PASSWORD = bytes(self.__PASSWORD, 'utf-8')

        # root
        self.SCHEME = 'https'
        self.HOST_NAME = 'dolphin.jump-technology.com'
        self.PORT = 8443
        self.APIVERSION = 'v1'
        self.URL = f'{self.SCHEME}://{self.HOST_NAME}:{self.PORT}/api/{self.APIVERSION}/'
        # self.HEADERS = { 'Authorization': self.__getAuth() }
        # self.session = Session(auth=(self.__USERNAME,self.__PASSWORD), headers=self.HEADERS)

        # session
        self.session = Session()
        self.session.auth = (self.__USERNAME, self.__PASSWORD)
        self.session.verify = False

    def get(self, endpoint, params):
        url = urljoin(self.URL, endpoint)
        req = self.session.get(url, params=params)
        req.raise_for_status()
        return req

    def post(self, endpoint, params, payload):
        url = urljoin(self.URL, endpoint)
        req = self.session.post(url, params=params, json=payload)
        req.raise_for_status()
        return req

    def put(self, endpoint, params, payload):
        url = urljoin(self.URL, endpoint)
        req = self.session.put(url, params=params, json=payload)
        req.raise_for_status()
        return req

    def checkpassword(self):
        return bytes(input("Password ?"), 'utf-8') == self.__PASSWORD

    '''
    def __getAuth(self):
        return 'Basic ' + b64encode(b':'.join((self.__USERNAME, self.__PASSWORD))).strip().decode('utf-8')
    '''


class RestManager(NetworkManager):
    '''
    High level RestManager class exposing main Dolphin API methods
    '''

    def __init__(self):
        super(RestManager, self).__init__()

    def getAssetList(self, parDate, columns=['ASSET_DATABASE_ID', 'LABEL', 'LAST_CLOSE_VALUE_IN_CURR', 'TYPE', 'CURRENCY'], fullResponse=False):
        '''
        Retourne la liste complète des actifs.
        Avec les valeurs des colonnes "ASSET_DATABASE_ID", "LABEL", "LABEL_CLOSE_VALUE_IN_CURR", "TYPE", "CURRENCY" à parDate.

                Parameters:
                        parDate (str): an RFC 3339 compliant date string
                        columns (list): the list of parameters
                        fullResponse (bool): wether to return the full response or not

                Returns:
                        json (list): list of pars dictionary
        '''
        params = {'date': parDate, 'columns': columns}
        if fullResponse:
            params['fullResponse'] = fullResponse
        resp = self.get('asset', params=params)
        return resp.json()

    def getAsset(self, parId, parDate, columns=['ASSET_DATABASE_ID', 'LABEL', 'LAST_CLOSE_VALUE_IN_CURR', 'TYPE', 'CURRENCY'], fullResponse=False):
        params = {'date': parDate, 'columns': columns}
        if fullResponse:
            params['fullResponse'] = fullResponse
        resp = self.get(f'asset/{parId}', params=params)
        return resp.json()

    def getAssetAttr(self, parId, parAttr, parDate, fullResponse=False):
        params = {'date': parDate}
        if fullResponse:
            params['fullResponse'] = fullResponse
        resp = self.get(f'asset/{parId}/attribute/{parAttr}', params=params)
        return resp.json()

    def getQuote(self, parId, parStartDate, parEndDate):
        params = {'start_date': parStartDate, 'end_date': parEndDate}
        resp = self.get(f'asset/{parId}/quote', params=params)
        return resp.json()

    def getPortfolio(self, parId):
        params = []
        resp = self.get(f'portfolio/{parId}/dyn_amount_compo', params=params)
        return resp.json()

    def putPortfolio(self, parId, portfolioLabel, currency, amount_type, values):
        params = []
        payload = {'label': portfolioLabel,
                   'currency': currency,
                   'type': amount_type,
                   'values': values}
        resp = self.put(
            f'portfolio/{parId}/dyn_amount_compo', params=params, payload=payload)
        return resp  # .json()

    def getRatio(self):
        params = []
        resp = self.get('ratio', params=params)
        return resp.json()

    def putRatio(self, ratio, asset, benchmark, start_date, end_date, frequency, fullResponse=False):
        params = []
        if fullResponse:
            params['fullResponse'] = fullResponse
        payload = {'ratio': ratio,
                   'asset': asset,
                   'benchmark': benchmark,
                   'start_date': start_date,
                   'end_date': end_date,
                   'frequency': frequency}
        resp = self.post('ratio/invoke', params=params, payload=payload)
        return resp.json()

    def getConvRate(self, currency_src, currency_dest="EUR"):
        req = f'currency/rate/{currency_src}/to/{currency_dest}'
        resp = self.get(req, params=[])
        return float(resp.json()["rate"]["value"].replace(",", "."))


if __name__ == "__main__":
    r = RestManager()
    ratios = r.getRatio()
    list_curs = ["USD", "EUR"]
    for cur in list_curs:
        print(r.getConvRate(cur))
    ratios = pd.DataFrame.from_dict(ratios).set_index("id")
    print(ratios)
