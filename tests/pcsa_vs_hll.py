#%%
'''
    Simulates HLL insertions and evaluate the expected counter value after N inserted elements
'''
import sys
sys.path.append("..")
sys.path.append(".")

from algorithms.pcsa import PCSA
from algorithms.hll import HyperLogLog
import string
import random
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import trange

#%% ----------- simulation parameters --------------
M = [512]
N = [100000] #np.logspace(3,5,10).astype(int)
charnum = 10
# -------------------------------------------
res = []

for m in M:
    for n in N:
        
        pcsa = PCSA(m)
        hll = HyperLogLog(m)
        true = set()

        hll_err = []
        pcsa_err = []

        samples = [i for i in range(0,n,100)]
        j = 0
        for i in trange(n):
            s = "".join([string.ascii_letters[random.randint(0, 
                        len(string.ascii_letters)-1)] for x in range(charnum)])
            hll.add(s)
            pcsa.add(s)
            true.add(s)

            if j < len(samples) and samples[j] == i:
                hll_err.append(100* (len(hll) - len(true))/len(true))
                pcsa_err.append(100* (len(pcsa) - len(true))/len(true))
                j += 1
        
        df1 = pd.DataFrame({'m': m, 
                            'N': n, 
                            'alg': 'HLL',
                             'err' : hll_err, 
                             'item': samples})
        df2 = pd.DataFrame({'m': m, 
                            'N': n, 
                            'alg': 'PCSA', 
                            'err' : pcsa_err, 
                            'item': samples})
        df = pd.concat([df1,df2], ignore_index=True)
        print('Generated unique', len(true))
#df.plot.line(y='err')
sns.lineplot(data=df, x='item', y='err', hue='alg')
plt.ylim([-30, 30])
plt.xlabel('items')
plt.ylabel('Estimation error [%]')
plt.axhline(0, color='k')
plt.show()