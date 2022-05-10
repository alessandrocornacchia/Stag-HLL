from algorithms.hll import *
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
            'HLLmle' : HyperLogLogMle,
            'HLLwM' : HyperLogLogWithPastMemory,
            'AHLL': AndreaTimeLogLog, 
            'StaggeredHLL': StaggeredHyperLogLog,
            'SlidingHLL' : SlidingHyperLogLog,
            'StaggeredHLL-vc' : StaggeredHyperLogLog}
    

def build_hll(name, W, m):
    Hll = hll_algos[name]
    if name in ['HLLwM', 'AHLL', 'StaggeredHLL', 'SlidingHLL', 'StaggeredHLL-vc']:
        if name == 'StaggeredHLL':    
            hll = Hll(W=W, m=m) #, hashf=(random_uniform_32bit,32))
        elif name == 'StaggeredHLL-vc':
            gamma = 0.875
            hll = Hll(W=W, m=m, mq=math.floor(gamma * m))
    else:                            
        hll = Hll(m=m)
    return hll
    
