import sys
sys.path.append("..")
sys.path.append(".")

from algorithms.pcsa import PCSA
from algorithms.hll import HyperLogLog
import string
import random

pcsa = PCSA(64)
hll = HyperLogLog(128)

for n in range(10000):
        #s = "".join([string.ascii_letters[random.randint(0, len(string.ascii_letters)-1)] for n in range(10)])
        s = str(n)
        pcsa.add(s)
        hll.add(s)
    
print(f'PCSA: {len(pcsa)}')
print(f'HLL: {len(hll)}')
#pcsa.dump()