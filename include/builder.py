from algorithms.hll import *
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
            'Stag-HLL': StaggeredHyperLogLog}
    

def build_hll(name, W, m):
    Hll = hll_algos[name]
    if name in ['HLLwM', 'AHLL', 'Stag-HLL']:
        hll = Hll(W=W, m=m) #, hashf=(random_uniform_32bit,32))
    else:                            
        hll = Hll(m=m)
    return hll
    
