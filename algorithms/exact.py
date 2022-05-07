from bisect import bisect_left
from itertools import groupby
from global_ import Runtime

'''
    - fast removal of outdated items
    - fast insertion of new items
    - fast cardinality query
'''
class StreamCardinalityCounter:

    __slots__ = ('W', 'history')
    _gettime = lambda x: x[0]
    _getid = lambda x: x[1]
        
    def __init__(self, W) -> None:
        self.W = W
        self.history = []

    '''
        insert new item : append at the end of the list
        N.B history will be sorted by construction
    '''
    def add(self, item, t=None):
        if t is None:
            t = Runtime.get().now
        # insert new item
        self.history.append((t,item))

    '''
        - binary search log(n) to locate timestamps outside W
        - unique to estimate cardinality
    '''
    def card(self, t=None):

        # if time not provided use simulation time
        if t is None:
            t = Runtime.get().now
        # index of oldest element in time window
        l = 0
        # binary search
        if t > self.W:
            t0 = t - self.W
            l = bisect_left(self.history, t0, key=StreamCardinalityCounter._gettime)
        
        # forget items outside window
        self.history = self.history[l:]
        
        # count uniques of remaining items
        uniq = set(map(StreamCardinalityCounter._getid, self.history))
        return len(uniq)


    def __len__(self):
        return self.card()
