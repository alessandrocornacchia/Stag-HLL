from algorithms.hll import *
from algorithms.pcsa import SlidingPCSA, SlidingPCSAPlus, StaggeredPCSA
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
            'SlidingPCSA' : SlidingPCSA,
            'SlidingPCSAPlus' : SlidingPCSAPlus}
    

def build_hll(name, W, m, b=32):
    Hll = hll_algos[name]
    if name == 'SlidingPCSAPlus':
        return Hll(W=W, m=m, tbits=b)
    elif name == 'StaggeredHLL-vc':
            gamma = 0.875
            return Hll(W=W, m=m, mq=math.floor(gamma * m))
    elif 'Staggered' in name or 'Sliding' in name:    
        return Hll(W=W, m=m) #, hashf=(random_uniform_32bit,32))
    else:                            
        return Hll(m=m)
    
    
