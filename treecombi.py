from string import ascii_uppercase as ASCIIUP
import time
import os
import random
import math
import pandas as pd
import numpy as np
from copy import deepcopy as dc
from network import RestManager
from portfolio import Portfolio

OKGREEN = '\033[92m'
OKYELLOW = '\033[33m'
ENDC = '\033[0m'


def rot(elems, n):
    return elems[n:] + elems[:n]


def get_combis(elems):  # elems = deque(elems)
    yield elems
    for n in range(1, len(elems)):  # elems.rotate(1)
        elems = rot(elems, 1)
        yield elems


class TreeCombi:
    # TODO: change nb_assets to 40
    def __init__(self, port, min_assets=15, max_assets=40,
                 lr=0.01, min_sharpe_change=0.05):
        # Data
        self.port = port
        self.end = len(port)
        self.elems = list(range(self.end))
        self.min_assets = min_assets
        self.max_assets = max_assets
        # Best
        self.bestcompo = ""
        self.bestsharpe = 0.0
        # Iter
        # self.nb_combis = pow(2, self.end) - 1  # this is wrong since min_assets
        # Following is also wrong since min_sharpe_change
        self.nb_combis = (pow(2, self.max_assets) - 1) * self.end
        self.iter = 0
        self.lr = lr
        self.min_sharpe_change = min_sharpe_change
        # Time
        self.t0 = time.time()
        self.t1 = time.time()
        self.elp = 0
        self.est = 0
        print(f'nb_combis: {self.nb_combis}')  # Wrong TODO: FIXME

    def __process(self, index):
        # FIXME: compute sharpe for port
        # get_sharpe(self.port)
        portSharp = self.port.get_sharpe()  # fake_sharpefunction(self.port)
        portSharp_1 = portSharp
        while True:
            # save navPer and portSharp
            navPer = self.port.dataframe.at[self.port.dataframe.index[index],
                                            "NAVPercentage"]
            portSharp = portSharp_1
            if math.isnan(portSharp):  # portSharp = float("-inf")
                raise RuntimeError(f"portSharp: {portSharp} is NAN !")
            # update df navPer according to learning rate self.lr
            # self.port[index] += self.lr
            if self.port.dataframe.at[self.port.dataframe.index[index],
                                      "NAVPercentage"] + self.lr >= 0.1:
                return portSharp
            self.port.dataframe.at[self.port.dataframe.index[index],
                                   "NAVPercentage"] = navPer + self.lr
            # compute new portShrap
            # fake_sharpefunction(self.port)
            portSharp_1 = self.port.get_sharpe()
            if portSharp_1 <= portSharp:
                break
        self.port.dataframe.at[self.port.dataframe.index[index],
                               "NAVPercentage"] = navPer
        return portSharp

    def __checkbetter(self, s, portsharpe):
        if portsharpe > self.bestsharpe:
            print(
                f'{OKGREEN}Better sharpe: {portsharpe: .2f},{OKYELLOW}+{portsharpe - self.bestsharpe: .2f}{ENDC}')
            self.bestsharpe = portsharpe
            self.bestcompo = s

    def __call__(self, start=-1, nb_assets=0, s=''):
        """
        TODO: we should handle index "rotations" in the start == -1 init loop,
        to have full trees each time but keeping the
        start > self.nb_assets condition
        if start > self.nb_assets:  # We do not take more than nb_assets assets
            return
        """
        if nb_assets >= self.max_assets:  # We need the rot approach
            return
        if start != -1:  # Process element at start index
            # str(self.port[start]) + ","  # Just for testing purpose
            nb_assets += 1
            s += str(self.c[start]) + ","
            self.iter += 1
            ti = time.time()  # took: Current iteration time

            # Processing
            # time.sleep(0.1 + random.random() / 2)  # Process element...
            portsharpe = self.__process(self.c[start])
            print(self.c[start])
            if nb_assets >= self.min_assets and nb_assets <= self.max_assets:
                assert(portsharpe == self.port.get_sharpe())
                self.__checkbetter(s, portsharpe)
                print(s, " - ", portsharpe)

            # Time & ouptut
            self.t1 = time.time()
            self.elp = self.t1 - self.t0  # cur_tm: total elapsed time
            # est_tm: total estimated time
            self.est = (self.elp / self.iter) * self.nb_combis
            print(f"iter {self.iter}/{self.nb_combis}: took: {self.t1 - ti: .1f}, cur_tm: {self.elp: .1f}, est_tm: {self.est: .1f}, left_tm: {self.est - self.elp: .1f}, bestsharpe:{self.bestsharpe}")
            if portsharpe > self.bestsharpe + self.min_sharpe_change:
                for i in range(start + 1, self.end):
                    port = self.port.dataframe["NAVPercentage"].copy()
                    self(i, nb_assets=nb_assets, s=s)
                    self.port.dataframe["NAVPercentage"] = port
        else:
            self.t0 = time.time()
            elems = dc(self.elems)
            for c in get_combis(elems):
                self.c = c  # Our elems rotation
                self(0, nb_assets=nb_assets, s=s)

    def set_default_valid_navs(self):
        best_stock_sharpes = self.port.dataframe[self.port.dataframe['assetType'] == 'STOCK']['sharpe'].copy(
        ).sort_values(ascending=False)  # by=['sharpe'],
        nb_asset = self.min_assets
        self.port.dataframe['NAVPercentage'] = 0.0
        for index in best_stock_sharpes.index[:nb_asset]:
            self.port.dataframe[index, 'NAVPercentage'] = 1.0 / nb_asset
        # for index in best_stock_sharpes.index[nb_asset:]:
        #    self.port.dataframe[index, 'NAVPercentage'] = 0.0

    def _markov(self, max_iter, percent_transfer, alpha):
        percentages = self.port.dataframe['NAVPercentage']
        min_nav_percentage = 0.01
        max_nav_percentage = 0.1
        current_sharpe = self.port.get_sharpe()
        for iteration in range(max_iter):
            reducable_percentages = percentages[percentages >= (
                min_nav_percentage + percent_transfer)]
            growable_percentages = percentages[percentages <= (
                max_nav_percentage - perccent_transfer)]
            source = reducable_percantages.index.sample(1)
            dest = growable_percentages.index.sample(1)

            percentages[source, 'NAVPercentage'] -= percent_transfer
            percentages[dest, 'NAVPercentage'] += percent_transfer

            if (not(self.port.is_valid())):
                # Rollback
                percentages[source, 'NAVPercentage'] += percent_transfer
                percentages[dest, 'NAVPercentage'] -= percent_transfer
            else:
                new_sharpe = self.port.get_sharpe()
                if new_sharpe > current_sharpe:
                    current_sharpe = new_sharpe
                else:
                    # Rollback
                    percentages[source, 'NAVPercentage'] += percent_transfer
                    percentages[dest, 'NAVPercentage'] -= percent_transfer

            percent_transfer *= alpha

    def markov(self, max_iter=1000, percent_transfer=0.02, alpha=1.0):
        self.set_default_valid_navs()
        assert(self.port.is_valid())
        self._markov(max_iter, percent_transfer, alpha)


if __name__ == "__main__":
    r = RestManager()
    # """
    if os.path.isfile("full.csv"):
        p = Portfolio(path="full.csv", restManager=r)
    else:
        p = Portfolio(restManager=r)
        df = p.dataframe.copy().sort_values(
            by=['sharpe'], ascending=False)  # [:5]
        p = Portfolio(dataframe=df, restManager=r)
        p.dump_portfolio()
    print(p.dataframe.columns)
    print(p.dataframe)  # ["sharpe"]
    # p.dataframe["NAVPercentage"] = 1.0 / p.dataframe.shape[0]
    print(p.init_correlation())
    if input("Launch ?") != "y":
        exit(0)
    p.dump_cov()
    t = TreeCombi(p)
    # t()
    t.markov()
    t.set_default_valid_navs()
    print(t.port.dataframe)
    print(t.port.get_sharpe())
