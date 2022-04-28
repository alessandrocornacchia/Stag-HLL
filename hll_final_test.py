from algorithms import  *
from algorithms.hll import HyperLogLog
from include.helpers import read_trace

f = './traces/caida1819/original/equinix-nyc.dirA.20180517-140000.UTC.anon/head_1M.csv'
hll = HyperLogLog(m=512)

stream = read_trace(f)
for t,f in stream:
    #print(f)
    hll.add(f)

print(len(hll))

