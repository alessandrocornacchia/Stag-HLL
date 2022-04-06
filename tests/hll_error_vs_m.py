'''
    Simulate stream of random 30-char strings.
    Evaluate HLL estimated cardinality for different parameters
        - m : the number of registers 
        - n : stream size

    TODO : multi-threading to run simulations in parallel
'''

from include.hyperloglogs.hll import HyperLogLog, HyperLogLogWithLinearCounting
import pandas as pd
import numpy as np
import string
import random
import signal
from tqdm import trange
import multiprocessing as mp

# ----------- global parameters --------------
m = [64, 128, 512]
N = [100] #, 1000, 10000, 100000]
R = 1000
charnum = 30
# -------------------------------------------

def worker(q,L,nameStr):
  
  while not q.empty():
    item = q.get()
    
    r = item['r']
    m = item['m']
    n = item['n']
    charnum = item['charnum']

    print(f'{nameStr}: seed={r}, m={m}, n={n}')
    
    hll = HyperLogLogWithLinearCounting(m)
    
    for i in range(n):
        s = "".join([string.ascii_letters[random.randint(0, 
                    len(string.ascii_letters)-1)] for x in range(charnum)])
        hll.add(s)
    
    E_Xi = np.mean(hll.M)
    tuple = (m, n, float(n - len(hll))/n, E_Xi)
    L.append(tuple)


# -----
if __name__ == '__main__':
    
    with mp.Manager() as manager:
        
        num_proc = mp.cpu_count()
        procs = []
        L = manager.list()  # <-- result list shared between processes.
        q = mp.Queue()

        for i in range(num_proc):
            nameStr = 'ID_'+str(i)
            p = mp.Process(target=worker, args=(q,L,nameStr,))
            procs.append(p)
        
        for r in range(R):
            np.random.seed(r)
            for mi in m:
                for n in N:
                    params = {'m': mi, 'r': r, 'n': n, 'charnum': charnum}
                    q.put(params)

    for j in procs:
        j.start()
    for j in procs:
        j.join()
    
    print(L)
    df = pd.DataFrame(L, columns=['m', 'N', 'err', 'E_xi'])
    df.to_csv('results/hll_counter_expectation.csv')