'''
    Simulates HLL insertions and evaluate the expected counter value after N inserted elements
'''
from include.hyperloglogs.hll import HyperLogLog, HyperLogLogWithLinearCounting,HyperLogLogExponential, HyperLogLogInverted
import pandas as pd
import numpy as np
import string
import random
from tqdm import trange


# ----------- simulation parameters --------------
M = [512]
N = [2**16] #np.logspace(3,5,10).astype(int)
R = 30
charnum = 30
# -------------------------------------------
res = []
for r in trange(R):
    
    np.random.seed(r)
    
    for m in M:
        for n in N:
    
            #hll = HyperLogLogInverted(m)
            hll = HyperLogLog(m)
            #hll = HyperLogLogWithLinearCounting(m)
            #hll = HyperLogLogExponential(m)
            
            for i in range(n):
                s = "".join([string.ascii_letters[random.randint(0, 
                            len(string.ascii_letters)-1)] for x in range(charnum)])
                hll.add(s)
            
            E_Xi = np.mean(hll.M)
            row = [m, n, float(n - len(hll))/n, E_Xi]
            res.append(row)

df = pd.DataFrame(res, columns=['m', 'N', 'err', 'E_xi'])
df.to_csv('results/hll_error_power_then_harmonic.csv')

#%% plot results
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% expected register value
EulerGamma = 0.57721
df = pd.read_csv('../Simulator/results/hll_counter_expectation.csv')
df['log2gamma'] = df[['m', 'N']].apply(lambda x: 
                        EulerGamma/np.log(2) + np.log2(float(x['N'])/x['m']), axis=1)

df['log2'] = df[['m', 'N']].apply(lambda x: np.log2(float(x['N'])/x['m']), axis=1)

'''
df['m'] = df['m'].astype(str)
sns.lineplot(x='N', y='E_xi', 
            hue='m', style="m", 
            markers=True, ci=95, 
            data=df, palette="Dark2")
plt.xscale('log')
'''
T = df.groupby(['m','N'])[['E_xi', 'log2gamma', 'log2']].mean().sort_values(['m', 'N'])
T

#%% distribution of the error
df = pd.read_csv('../Simulator/results/hll_error_harmonic_then_power.csv')
sns.distplot(x='N', y='err', data=df, palette="Dark2")