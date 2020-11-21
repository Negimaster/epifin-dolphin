from string import ascii_uppercase as ASCIIUP
import time
import random
import numpy as np


def fake_sharpefunction(elms):
    return -np.std(elms)


class TreeCombi:
    def __init__(self, elems, lr=0.01):
        # Data
        self.elems = elems
        self.end = len(elems)
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
        portSharp = fake_sharpefunction(self.elems)
        portSharp_1 = portSharp
        while True:
            # save navPer and portSharp
            navPer = self.elems[index]
            portSharp = portSharp_1
            # update df navPer according to learning rate self.lr
            self.elems[index] += self.lr
            # compute new portShrap
            portSharp_1 = fake_sharpefunction(self.elems)
            if portSharp_1 <= portSharp:
                break
        self.elems[index] = navPer
        print(portSharp)

    def __call__(self, start=-1, s=''):
        if start != -1:  # Process element at start index
            # str(self.elems[start]) + ","  # Just for testing purpose
            s += str(start) + ","
            self.iter += 1
            ti = time.time()  # took: Current iteration time

            # Processing
            # time.sleep(0.1 + random.random() / 2)  # Process element...
            self.__process(start)
            print(s, " - ", self.elems)

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
    X = np.array(list(range(4))).astype("float")
    print(X)
    #t = TreeCombi(list(ASCIIUP)[:4])
    t = TreeCombi(X)
    t()
