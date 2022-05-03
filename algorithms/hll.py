"""
This module implements probabilistic data structure which is able to calculate the cardinality of large multisets in a single pass using little auxiliary memory
"""

from argparse import ArgumentError
from bisect import bisect_left
import math
import numpy as np
from include.hashfunctions import sha1_32bit
import warnings
import logging
import sympy
import copy
from global_ import Runtime

def bit_length(w):
    return w.bit_length()

def get_alpha(p):
    if not (4 <= p <= 16):
        raise ValueError("p=%d should be in range [4 : 16]" % p)

    if p == 4:
        return 0.673

    if p == 5:
        return 0.697

    if p == 6:
        return 0.709

    return 0.7213 / (1.0 + 1.079 / (1 << p))

def get_rho(w, max_width):
    rho = max_width - bit_length(w) + 1

    if rho <= 0:
        raise ValueError('w overflow')

    return rho

'''
    Harmonic mean of elements in x
'''
def harmonic_mean(x):
    m = len(x)
    return m / np.sum(1/x)


'''
    HyperLogLog without small and large range corrections
'''
class HyperLogLogNoCorrections(object):
    
    __slots__ = ('p', 'm', 'M', 'hash_function', 'hashbit', 'alpha', 'n')

    """
    HyperLogLog cardinality estimator

        # error_rate = abs_err / true cardinality
        # m = 2 ** p [number of register]
        # error_rate = 1.04 / sqrt(m)
        # registers M(1)... M(m)
    """
    def __init__(self, m=None, error_rate=None, hashf=(sha1_32bit, 32)):
        
        if error_rate is not None and not (0 < error_rate < 1):
            raise ValueError("Error_Rate must be between 0 and 1.")

        if m is None and error_rate is not None:
            p = int(math.ceil(math.log((1.04 / error_rate) ** 2, 2)))
            self.m = 1 << p                         # number of registers
        elif m is not None:
            p = int(np.ceil(np.log2(m)))
            self.m = m
        else:
            raise ArgumentError("Specify either m or error_rate")
        
        
        self.alpha = get_alpha(p)    
        self.p = p                              # number of bits used to index substream
        self.M = np.zeros(self.m, dtype=int)    # registers
        self.n = np.zeros(self.m, dtype=int)    # items/register (for statistics)
        self.hash_function = hashf[0]           # hash function to be used
        self.hashbit = hashf[1]                 # number of bits of hash function

    def __getstate__(self):
        return dict([x, getattr(self, x)] for x in self.__slots__)

    def __setstate__(self, d):
        for key in d:
            setattr(self, key, d[key])

    ''' 
        Extracts register id and rank from hash bitstring 
    '''
    def _reg_and_rank(self, value):
        x = self.hash_function(value)
        logging.debug(f'hash: {x}')
        j = x & (self.m - 1)
        w = x >> self.p
        return (j, get_rho(w, self.hashbit - self.p))

    def add(self, value):
        """
        Adds the item to the HyperLogLog
            - h: D -> {0,1} ** 32
            - x = h(v)
            - j = <x_0x_1..x_{p-1}>
            - w = <x_{p}x_{p+1}..>
            - M[j] = max(M[j], rho(w))
        """
        (j,rho) = self._reg_and_rank(value)
        self.M[j] = max(self.M[j], rho)
        self.n[j] += 1

    def update(self, *others):
        """
        Merge other counters
        """
        for item in others:
            if self.m != item.m:
                raise ValueError('Counters precisions should be equal')
        self.M = [max(*items) for items in zip(*([ item.M for item in others ] + [ self.M ]))]

    def __eq__(self, other):
        if self.m != other.m:
            raise ValueError('Counters precisions should be equal')
        return self.M == other.M

    def __ne__(self, other):
        return not self.__eq__(other)

    def __len__(self):
        return round(self.card())

    def __str__(self) -> str:
        return "biasHLL-m%d" % self.m

    def card(self):
        """
        Returns the estimate of the cardinality
        """
        return self.alpha * self.m * harmonic_mean(2**self.M)
            

'''
    Flajolet HyperLogLog with small and large range corrections
    as in original paper
'''
class HyperLogLog(HyperLogLogNoCorrections):
    
    __slots__ = ('_small_range_threshold', '_large_range_threshold')

    def __init__(self, m=None, error_rate=None, hashf=(sha1_32bit, 32)):
        super().__init__(m, error_rate, hashf)
        self._small_range_threshold = (5.0 / 2.0) * self.m
        self._large_range_threshold = (1.0 / 30.0) * (1 << self.hashbit)
        
    def __str__(self) -> str:
        return "HLL-m%d" % self.m

    def _linearcounting(self, V):
        # count number or registers equal to 0
        logging.debug(f'linear counting estimate: {self.m * np.log(self.m / V)}')
        return self.m * np.log(self.m / V)

    def _largerange_correction(self, E):
        return - (1 << self.hashbit) * np.log(1.0 - E / (1 << self.hashbit))

    def card(self):
        """
        Returns the estimate of the cardinality
        """
        # HLL estimate
        E = self.alpha * self.m * harmonic_mean(2**self.M)
        
        if abs(E-self._small_range_threshold)/self._small_range_threshold < 0.15:
          warnings.warn(("Warning: estimate is close to small error correction threshold. "
                        +"Output may not satisfy HyperLogLog accuracy guarantee."))
        # Small range correction i.e. resort to linear counting
        V = (self.M == 0).sum() # number of registers equal zero
        if V > 0 and E <= self._small_range_threshold:
            return self._linearcounting(V)
        # Normal range, no correction
        if E <= self._large_range_threshold:
            return E
        # Large range correction
        return self._largerange_correction(E)

''' Represents a temporal HLL without memory restrictions
i.e. capable of mantaining history of the past W seconds''' 
class HyperLogLogWithPastMemory(HyperLogLog):
    
    __slots__ = ('W', 'history')

    def __init__(self, W, m=None, error_rate=None, hashf=(sha1_32bit, 32)):
        super().__init__(m, error_rate, hashf)
        self.history = []
        self.W = W

    def __str__(self) -> str:
        return "HLL-ref-m%d" % self.m

    def add(self, value):
        t = Runtime.get().now
        self.history.append((t,value))

    def card(self):
        # remove outdated elements from the past history. The list
        # is ordered by constuction, we can apply binary search
        t = Runtime.get().now
        if t > self.W:
            l = bisect_left(self.history, t - self.W, key=lambda x: x[0])
        else:
            l = 0
        self.history = self.history[l:]
        
        # now re-process the substream with hll
        # i.e. reset counters and re-fill HLL only with recent history
        # n.b use super().add() to add as in standard hyperloglog
        self.M = np.zeros(self.m) #[0 for m in self.M]
        for _, item in self.history:
            super().add(item)
        # then return usual cardinality estimator
        return super().card()
        

''' 
    Like HLL but for cardinality query first compute H, harmonic mean of the
    registers, then return C = 2**H.
'''
class HyperLogLogInverted(HyperLogLogNoCorrections):

    def __str__(self) -> str:
        return "HLL-inv-m%d" % self.m

    def card(self):
        """
        Returns the estimate of the cardinality
        """
        x = self.M[self.M != 0]
        r = harmonic_mean(x)
        return self.m * 2**r


''' exponential rank, hash split ''' 
class HyperLogLogExponential(HyperLogLog):

    __slots__ = ('gamma')    

    ''' same as HLL but with continuous counters '''
    def __init__(self, m=None, error_rate=None, hashf=(sha1_32bit, 32)):
        super().__init__(m, error_rate, hashf)
        self.M = np.zeros(self.m)
        self.gamma = float(sympy.EulerGamma.evalf())/np.log(2)

    
    ''' get ideal rank i.e. rank same as expected value'''
    def _reg_and_rank(self, value):
        # use hash to decide register index
        x = self.hash_function(value)
        j = x & (self.m - 1)  
        # rank generated with exponential law
        rho = np.random.exponential(1./np.log(2))
        return (j, rho)

    
    ''' cardinality estimation : de-bias and get estimate'''
    def card(self):
        return self.m * harmonic_mean(2**self.M)
    
    def __str__(self) -> str:
        return "HLLexp-m%d" % self.m


''' exponential rank, equal split '''
class HyperLogLogExponentialEqualSplit(HyperLogLogExponential):

    __slots__ = ('j')    

    def __init__(self, m=None, error_rate=None, hashf=(sha1_32bit, 32)):
        super().__init__(m, error_rate, hashf)
        self.j = 0

    ''' get exponential rank with ideal flow split '''
    def _reg_and_rank(self, value):
        # split uniformly incoming flows on registers
        self.j = (self.j + 1) % self.m    
        rho = np.random.exponential(1./np.log(2))
        return (self.j, rho)

    def __str__(self) -> str:
        return "HLLexpes-m%d" % self.m


'''manual rank, equal split '''
class HyperLogLogEqualSplit(HyperLogLogExponentialEqualSplit):

    def _reg_and_rank(self, value):
        # split uniformly incoming flows on registers
        self.j = (self.j + 1) % self.m    
        # rank to be exactly log2 of the observed items
        rho = self.gamma + np.log2(self.n[self.j]+1)         
        return (self.j, rho)


''' hll with MLE cardinality estimator '''
class HyperLogLogMle(HyperLogLogExponential):

    def card(self):
        n = - 1./np.log(1-np.exp(-np.log(2) * self.M))
        return float(self.m**2) / np.sum(1./n)

    def __str__(self) -> str:
        return "HLLmle-m%d" % self.m


'''
    Staggered HyperLogLog. Periodic register reset
'''
class StaggeredHyperLogLog(HyperLogLog):

    __slots__ = ('rst', 'k', 'c', 'W', 'to', 'l', 'eM', 'Mq')

    def __init__(self, W, m=None, error_rate=None, hashf=(sha1_32bit, 32)):
        super().__init__(m,error_rate,hashf)
        self.rst = 0
        self.W = W
        self.Mq = np.zeros(self.m)
        self.eM = np.zeros(self.m)                           # "equalized" registers
        self.c = self._compute_constants()                   # constants
        self.k = np.roll(np.arange(self.m-1, -1, -1), -1)    # register index 
        self.to = float(2 * self.W) / self.m                 # reset timeout
        logging.debug(f'initial configuration: {self.k}')
        logging.debug(f'equalization constants: {self.c}')
        logging.info(f'alpha: {self.alpha}, reset timeout: {self.to}')

    def __str__(self) -> str:
        return "stgHLL-m%d" % self.m

    # pre-compute constants for speed
    def _compute_constants(self):
        #EulerGamma = float(sympy.EulerGamma.evalf())
        i = np.arange(1, self.m+1)
        #return (np.log(self.m**2 / (2 * self.W * i)) - EulerGamma) / np.log(2) - 0.5
        return np.log2(self.m**2 / (2 * self.W * i))
        
    '''
        Executes reset of one HLL register, circularly
    '''
    def circular_reset(self):
        # hold register content at last reset time
        self.Mq = copy.copy(self.M) 
        # reset  register content
        self.M[self.rst] = 0
        self.n[self.rst] = 0
        self.rst = (self.rst + 1) % self.m

    """
    Returns the estimate of the cardinality
    """
    def card(self, t=None):
        # ask current simulation time if simulating
        if t is None:
            t = Runtime.get().now
        
        # acquire phase-synchronization 
        offset = math.floor(t/self.to)
        k = (offset + self.k) % self.m
        
        # register equalization
        self.eM = self.Mq + self.c[k]

        # estimation as in Flajolet
        E = self.alpha * self.W * harmonic_mean(2 ** self.eM)

        V = (self.Mq == 0).sum()
        if V > 0 and E <= self._small_range_threshold:
            E = self._linearcounting(V)

        logging.debug(f'[t={t:.5f} s] phase-offset: {offset}')
        logging.debug(f'[t={t:.5f} s] i: {k}')
        logging.debug(f'[t={t:.5f} s] M: {self.M}')
        logging.debug(f'[t={t:.5f} s] c: {self.c[k]}')
        logging.debug(f'[t={t:.5f} s] M equalized: {self.eM}')
        logging.debug(f'[t={t:.5f} s] Estimation: {E}')

        return E


'''
    AndreaTimeLogLog : smoothing of counter values
'''
class AndreaTimeLogLog(HyperLogLog):
    
    __slots__ = ('Ts', 'Mmax', 'W')

    def __init__(self, W, m=None, error_rate=None, hashf=(sha1_32bit, 32)):
        super().__init__(m,error_rate,hashf)
        self.Ts = [ 0 for i in range(self.m) ] # timestamp
        self.Mmax = [ 0 for i in range(self.m) ] # current maximum rank
        self.W = W

    def __str__(self) -> str:
        return "AndreaTimeLogLog-m%d" % self.m

    def add(self, value):
        
        time = Runtime.get().now
        (j,rho) = self._reg_and_rank(value)
        # 1. Update-current maximum based on elapsed time since last arrival
        elapsed = time - self.Ts[j]
        # if out of transient
        if time >= self.W:
            # if inside window span smooth current maximum, else apply some heursitic e.g. -1
            if elapsed <= self.W:
                self.M[j] = max(0, np.log2(float(self.W-elapsed)/self.W * 2**self.Mmax[j]))
            else:
                self.M[j] = self.Mmax[j] - 1
                self.Ts[j] = time
                self.Mmax[j] = self.M[j]
        # 2. Compare with item rank -> update if new maximum 
        if self.M[j] < rho:
            self.Ts[j] = time
            self.Mmax[j] = rho
        # log message
        log = f"t={time}, Ts={self.Ts[j]}, rho={rho}, M{j}(t)={self.M[j]}"
        logging.debug(log)




if __name__ == '__main__':
    import pickle
    import string
    import random 
    error = 1.04 / np.sqrt(1)
    hll = HyperLogLog(error, hashf=(lambda x: random.randint(0, 2**4 - 1), 4))
    for n in range(50):
        s = "".join([string.ascii_letters[random.randint(0, len(string.ascii_letters)-1)] for n in range(2)])
        hll.add(s)
        print('Adding %s - HLL %d - Estimation %d' % (s, hll.M[0], len(hll)))
        