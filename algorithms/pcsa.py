"""
This module implements PCSA sketch, also known as Flajolet-Martin sketch
a probabilistic data structure which is able to calculate the cardinality of large multisets
in a single pass using little auxiliary memory
"""

from argparse import ArgumentError
from bisect import bisect_left
import math
import numpy as np
from include.hashfunctions import sha1_32bit
from .hll import get_rho
import warnings
import logging
import sympy
import copy
from global_ import Runtime

'''
    Represents timestamp with desired resolution (nbits)
'''
def round_up(t, nbits=32):
    dt = 2**(-nbits)             # time step
    fractional = t - int(t)      # fractional part
    return np.ceil(fractional/dt) * dt + int(t)

def round_down(t, nbits=32):
    dt = 2**(-nbits)             # time step
    fractional = t - int(t)      # fractional part
    return np.floor(fractional/dt) * dt + int(t)

def round_closest(t, nbits=32):
    dt = 2**(-nbits)             # time step
    fractional = t - int(t)      # fractional part
    return np.round(fractional/dt) * dt + int(t)

def round_random(t, nbits=32):
    dt = 2**(-nbits)             # time step
    fractional = t - int(t)      # fractional part
    p = np.random.uniform()
    if p < 0.5:
        return np.ceil(fractional/dt) * dt + int(t)
    else:
        return np.floor(fractional/dt) * dt + int(t)

'''
    Probabilistic Counting with Stochastic Averaging
    also known as Flajolet-Martin sketch (FM sketch)
'''
class PCSA(object):
    
    __slots__ = ('p', 'm', 'B', 'hash_function', 'hashbit')
    PHI = 0.77351
    CHI = 1.75

    """
    Probabilistic Counting cardinality estimator
    """
    def __init__(self, m=None, error_rate=None, hashf=(sha1_32bit, 32)):
        
        if error_rate is not None and not (0 < error_rate < 1):
            raise ValueError("Error_Rate must be between 0 and 1.")

        if m is None and error_rate is not None:
            p = int(math.ceil(math.log((0.78 / error_rate) ** 2, 2)))
            self.m = 1 << p                         # number of registers
        elif m is not None:
            p = int(np.ceil(np.log2(m)))
            self.m = m
        else:
            raise ArgumentError("Specify either m or error_rate")
           
        self.p = p                                  # number of bits used to index substream
        self.hash_function = hashf[0]               # hash function to be used
        self.hashbit = hashf[1]                     # number of bits of hash function
        self.B = np.zeros((self.m, self.hashbit - self.p), dtype=bool)  # bitmaps
        
        logging.debug(f'Initialized PCSA with m={self.m} bitmaps, of size {self.B.shape}')

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
        j = x & (self.m - 1)
        logging.debug(f'hash: {x}, substream: {j}')
        w = x >> self.p
        return (j, get_rho(w, self.hashbit - self.p))

    def add(self, value):
        """
        Adds the item to the PCSA
            - h: D -> {0,1} ** 32
            - x = h(v)
            - B[j][rho] = 1
        """
        (j,rho) = self._reg_and_rank(value)
        #self.B[j] = self.B[j] | np.uint32(2**31) >> (rho-1)
        self.B[j][rho-1] = 1

    def update(self, *others):
        """
        Merge other counters
        """
        raise NotImplementedError()

    def __eq__(self, other):
        if self.m != other.m:
            raise ValueError('Counters precisions should be equal')
        return self.B == other.B

    def __ne__(self, other):
        return not self.__eq__(other)

    def __len__(self):
        return round(self.card())

    def __str__(self) -> str:
        return "PCSA-m%d" % self.m

    def dump(self):
        for bitmap in self.B:
            print(''.join([f'{bit:1b}' for bit in bitmap]))


    def card(self):
        """
        Returns the estimate of the cardinality
            - Z : \sum i=1 ^ m \rho_i
            - \rho_i : position of the first least-significant 0 bit (right-most) in the bitmap
        """
        Z = np.sum([np.where(bitmap == False)[0][0] for bitmap in self.B]).astype(float)
        return self.m / PCSA.PHI  * ( 2 ** (Z/self.m) - 2 ** (- PCSA.CHI * Z/self.m) )
            

'''
    Staggered PCSA. Periodic register reset
'''
class StaggeredPCSA(PCSA):

    __slots__ = ('rst', 'k', 'c', 'W', 'to', 'l', 'Bq', 'rpp')

    def __init__(self, W, m=None, mq=None, error_rate=None, hashf=(sha1_32bit, 32)):
        super().__init__(m,error_rate,hashf)
        self.rst = 0
        self.W = W
        self.B = np.zeros((self.m, self.hashbit - self.p), dtype=bool)
        mq = self.m if mq is None else mq
        self.rpp = self.m - mq                          # register pruning index
        #self.c = self._compute_constants()                   # constants
        #self.k = np.roll(np.arange(self.m-1, -1, -1), -1)    # register index 
        self.to = float(2 * self.W) / self.m                 # reset timeout
        logging.debug(f'Staggered PCSA, m={self.m}, query enabled in range [{self.rpp}-{self.m-1}]')
        #logging.debug(f'initial configuration: {self.k}')
        #logging.debug(f'equalization constants: {self.c}')
        logging.info(f'reset timeout: {self.to}')

    def __str__(self) -> str:
        return "stgPCSA-m%d" % self.m

    # pre-compute constants for speed
    def _compute_constants(self):
        #EulerGamma = float(sympy.EulerGamma.evalf())
        return np.log2(self.m / self.W) * np.ones(self.m)
    
    '''
        Executes reset of one PCSA register, circularly
    '''
    def circular_reset(self):
        # hold register content at last reset time
        self.Bq = copy.copy(self.B) 
        # reset  register content
        self.B[self.rst] = np.zeros(self.hashbit - self.p, dtype=bool)
        # self.n[self.rst] = 0
        self.rst = (self.rst + 1) % self.m

    def card(self):
        """
        Returns the estimate of the cardinality
            - Z : \sum i=1 ^ m \rho_i
            - \rho_i : position of the first least-significant 0 bit (right-most) in the bitmap
        """
        Z = np.sum([np.where(bitmap == False)[0][0] for bitmap in self.Bq]).astype(float)
        return self.m / PCSA.PHI  * ( 2 ** (Z/self.m) - 2 ** (- PCSA.CHI * Z/self.m) )



class SlidingPCSA(PCSA):

    __slots__ = ('W', 'Brt')

    def __init__(self, W, m=None, error_rate=None, hashf=(sha1_32bit, 32)):
        super().__init__(m, error_rate, hashf)
        w = self.hashbit - self.p
        self.B = -np.inf * np.ones((self.m, w), dtype=float)  # bitmaps
        self.Brt = [[[] for _ in range(w)] for _ in range(self.m)]
        self.W = W
        

    def add(self, value):
        (j,rho) = self._reg_and_rank(value)
        t = Runtime.get().now
        # statistical purposes
        self.Brt[j][rho-1].append(t - self.B[j][rho-1])
        # update bit
        self.B[j][rho-1] = t
        

    def card(self):
        """
        Returns the estimate of the cardinality
            - Z : \sum i=1 ^ m \rho_i
            - \rho_i : position of the first least-significant 0 bit (right-most) in the bitmap
        """
        t = Runtime.get().now
        Z = np.sum([np.where(bitmap < t - self.W)[0][0] for bitmap in self.B]).astype(float)
        return self.m / PCSA.PHI  * ( 2 ** (Z/self.m) - 2 ** (- PCSA.CHI * Z/self.m) )



'''
    Sliding PCSA with offset trick
'''
class SlidingPCSAPlus(SlidingPCSA):

    __slots__ = ('b')

    def __init__(self, W, m=None, error_rate=None, tbits=32, hashf=(sha1_32bit, 32)):
        super().__init__(W, m=m, error_rate=error_rate, hashf=hashf)
        self.b = tbits
        

    def add(self, value):
        (j,rho) = self._reg_and_rank(value)
        t = Runtime.get().now
        # statistical purposes
        # self.Brt[j][rho-1].append(t - self.B[j][rho-1])
        
        # exponential increase every K bits
        #b = min(self.B.shape[1], 2 ** (np.ceil(rho / 2)-1))
        
        # linear increase every K bits
        #b = min(self.B.shape[1], np.ceil(rho / 2))
        
        self.B[j][rho-1] = round_closest(t, nbits=self.b) #round_closest(t, nbits=self.b)
        
        #logging.debug(f'rank: {rho}, bit allocation: {self.b}')


    def card(self):
        """
        Returns the estimate of the cardinality
            - Z : \sum i=1 ^ m \rho_i
            - \rho_i : position of the first least-significant 0 bit (right-most) in the bitmap
        """
        t = Runtime.get().now
        
        Z = 0
        for j in range(self.m):
            offset = (float(-self.m)/2 + j) * 2**(-self.b)/self.m
            offset = 0
            Z += np.where(self.B[j] + offset < t - self.W)[0][0]

        return self.m / PCSA.PHI  * ( 2 ** (Z/self.m) - 2 ** (- PCSA.CHI * Z/self.m) )


if __name__ == '__main__':
    
    import string
    import random 
    
    pcsa = PCSA(8)
    pcsa.add('B')

    for n in range(50):
        s = "".join([string.ascii_letters[random.randint(0, len(string.ascii_letters)-1)] for n in range(2)])
        hll.add(s)
        print('Adding %s - HLL %d - Estimation %d' % (s, hll.M[0], len(hll)))
        