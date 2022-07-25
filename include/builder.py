from algorithms.hll import *
from algorithms.pcsa import SlidingPCSA, StaggeredPCSA
from algorithms.shll import SlidingHyperLogLog
from include.hashfunctions import *

''' 
    ------ Supported algorithms must go here ------
    
    N.B. only algorithms available here will be recognized as available choices 
    in CLI (command line interface).
    -----------------------------------------------
    '''
hll_algos ={'HLL': HyperLogLog,
            'bHLL' : HyperLogLogNoCorrections,    
            'HLLe': HyperLogLogExponential,
            'HLLei': HyperLogLogExponentialEqualSplit,
            'HLLi' : HyperLogLogEqualSplit,
            'HLLwM' : HyperLogLogWithPastMemory,
            'StaggeredHLL': StaggeredHyperLogLog,
            'SlidingHLL' : SlidingHyperLogLog,
            'StaggeredHLL-vc' : StaggeredHyperLogLog,
            'StaggeredPCSA' : StaggeredPCSA,
            'SlidingPCSA' : SlidingPCSA}
    

def build_hll(name, W, m, b=32):
    Hll = hll_algos[name]
    if 'Staggered' in name or 'Sliding' in name:    
        if name == 'StaggeredHLL-vc':
            gamma = 0.875
            hll = Hll(W=W, m=m, mq=math.floor(gamma * m))
        hll = Hll(W=W, m=m, tbits=b) #, hashf=(random_uniform_32bit,32))
    else:                            
        hll = Hll(m=m)
    return hll
    
