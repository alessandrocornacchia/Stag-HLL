"""
Sliding HyperLogLog
"""

import math
import heapq
import struct
import numpy as np
import warnings
import logging
import sympy
from global_ import Runtime
from include.hashfunctions import sha1_32bit
from .hll import HyperLogLog, get_alpha, harmonic_mean


class SlidingHyperLogLog(HyperLogLog):
    """
    Sliding HyperLogLog: Estimating cardinality in a data stream (Telecom ParisTech)
    """

    __slots__ = ('W')

    def __init__(self, W, m=None, error_rate=None, hashf=(sha1_32bit, 32), lpfm=None):
      
        self.W = W

        if lpfm is not None:
            m = len(lpfm)
            p = int(round(math.log(m, 2)))

            if (1 << p) != m:
                raise ValueError('List length is not power of 2')
            self.M = lpfm

        else:
            if error_rate is not None and not (0 < error_rate < 1):
                raise ValueError("Error_Rate must be between 0 and 1.")

            if m is None and error_rate is not None:
                p = int(math.ceil(math.log((1.04 / error_rate) ** 2, 2)))
            elif m is not None:
                p = int(np.log2(m))
            else:
                raise ValueError("Specify either m or error_rate")

            m = 1 << p
            self.M = [None for i in range(m)]

        self.alpha = get_alpha(p)
        self.p = p
        self.m = m
        self.hash_function = hashf[0]           # hash function to be used
        self.hashbit = hashf[1]                 # number of bits of hash function
        self._small_range_threshold = (5.0 / 2.0) * self.m
        self._large_range_threshold = (1.0 / 30.0) * (1 << self.hashbit)

    @classmethod
    def from_list(cls, lpfm, W):
        return cls(None, W, lpfm)

    def add(self, value, timestamp=None):
        """
        Adds the item to the HyperLogLog
        
        # h: D -> {0,1} ** 64
        # x = h(v)
        # j = <x_0x_1..x_{p-1})>
        # w = <x_{p}x_{p+1}..>
        # <t_i, rho(w)>
        """
        if timestamp is None:
            timestamp = Runtime.get().now

        (j, R) = self._reg_and_rank(value)
        Rmax = None
        tmp = []
        tmax = None
        tmp2 = list(heapq.merge(self.M[j] if self.M[j] is not None else [], [(timestamp, R)]))

        for t, R in reversed(tmp2):
            if tmax is None:
                tmax = t

            if t < (tmax - self.W):
                break

            if Rmax is None or R > Rmax:
                tmp.append((t, R))
                Rmax = R

        tmp.reverse()
        self.M[j] = tuple(tmp) if tmp else None

    def update(self, *others):
        """
        Merge other counters
        
        for item in others:
            if self.m != item.m:
                raise ValueError('Counters precisions should be equal')

        for j in range(len(self.M)):
            Rmax = None
            tmp = []
            tmax = None
            tmp2 = list(heapq.merge(*([item.M[j] if item.M[j] is not None else [] for item in others] + [self.M[j] if self.M[j] is not None else []])))

            for t, R in reversed(tmp2):
                if tmax is None:
                    tmax = t

                if t < (tmax - self.W):
                    break

                if Rmax is None or R > Rmax:
                    tmp.append((t, R))
                    Rmax = R

            tmp.reverse()
            self.M[j] = tuple(tmp) if tmp else None
            """
        raise NotImplemented

    def card(self, timestamp=None):
        """
        Returns the estimate of the cardinality at 'timestamp' using 'W'
        """

        if timestamp is None:
            timestamp = Runtime.get().now

        def max_r(l):
            return max(l) if l else 0

        M = np.array([max_r([R for ts, R in lpfm if ts >= (timestamp - self.W)]) if lpfm else 0 for lpfm in self.M])

        # HyperLogLog harmonic mean estimate with registers populated with pkts in W
        E = self.alpha * self.m * harmonic_mean(2**M)

        # count number or registers equal to 0
        V = (M==0).sum()
        if V > 0 and E <= self._small_range_threshold:
            return self._linearcounting(V)
        # Normal range, no correction
        if E <= self._large_range_threshold:
            return E
        # Large range correction
        return self._largerange_correction(E)