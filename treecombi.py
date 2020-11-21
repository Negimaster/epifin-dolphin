from string import ascii_uppercase as ASCIIUP
import time
import os
import random
import pandas as pd
import numpy as np
from network import RestManager
from portfolio import Portfolio

"""
def fake_sharpefunction(elms):
    return -np.std(elms)
"""


class TreeCombi:
    def __init__(self, port, lr=0.01):
        # Data
        self.port = port
        self.end = len(port)
        # Iter
        self.nb_combis = pow(2, self.end) - 1
        self.iter = 0
        self.lr = lr
        # Time
        self.t0 = time.time()
        self.t1 = time.time()
        self.elp = 0
        self.est = 0
        print(f'nb_combis: {self.nb_combis}')

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
            # update df navPer according to learning rate self.lr
            # self.port[index] += self.lr
            self.port.dataframe.at[self.port.dataframe.index[index],
                                   "NAVPercentage"] = navPer + self.lr
            # compute new portShrap
            # fake_sharpefunction(self.port)
            portSharp_1 = self.port.get_sharpe()
            if portSharp_1 <= portSharp:
                break
        self.port.dataframe.at[self.port.dataframe.index[index],
                               "NAVPercentage"] = navPer
        print(portSharp)

    def __call__(self, start=-1, s=''):
        if start != -1:  # Process element at start index
            # str(self.port[start]) + ","  # Just for testing purpose
            s += str(start) + ","
            self.iter += 1
            ti = time.time()  # took: Current iteration time

            # Processing
            # time.sleep(0.1 + random.random() / 2)  # Process element...
            self.__process(start)
            print(s, " - ", self.port.get_sharpe())

            # Time & ouptut
            self.t1 = time.time()
            self.elp = self.t1 - self.t0  # cur_tm: total elapsed time
            # est_tm: total estimated time
            self.est = (self.elp / self.iter) * self.nb_combis
            print(f"iter {self.iter}/{self.nb_combis}: took: {self.t1 - ti: .1f}, cur_tm: {self.elp: .1f}, est_tm: {self.est: .1f}, left_tm: {self.est - self.elp: .1f}")
        else:
            self.t0 = time.time()
        for i in range(start + 1, self.end):
            self(i, s)


if __name__ == "__main__":
    """
    X = np.array(list(range(4))).astype("float")
    print(X)
    #t = TreeCombi(list(ASCIIUP)[:4])
    t = TreeCombi(X)
    t()
    """
    r = RestManager()
    if os.path.isfile("save20bestsharpe.csv"):
        df = pd.read_csv("save20bestsharpe.csv", index_col=0)
        df = df.astype({"totalValue": "float64", "NAVPercentage": "float64"})
        p = Portfolio(dataframe=df, restManager=r)
    else:
        p = Portfolio(restManager=r)
        df = p.dataframe.copy().sort_values(
            by=['sharpe'], ascending=False)[:20]
        p = Portfolio(dataframe=df, restManager=r)
        p.dataframe.to_csv("save20bestsharpe.csv")  # , index=False)
    print(p.dataframe.columns)
    print(p.dataframe)  # ["sharpe"]
    p.dataframe["NAVPercentage"] = 1.0 / p.dataframe.shape[0]
    t = TreeCombi(p)
    t()
    print(t.port.dataframe)
    print(t.port.get_sharpe())
