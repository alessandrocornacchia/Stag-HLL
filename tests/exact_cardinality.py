import sys
sys.path.append("..")
sys.path.append(".")

from algorithms.exact import StreamCardinalityCounter
from include.helpers import read_trace
from itertools import groupby

f = './traces/caida1819/original/equinix-nyc.dirA.20180517-140000.UTC.anon/head_100k.csv'
pkts = list(read_trace(f))
for _,f in pkts[:10]:
    print(f)

_, flows = zip(*pkts[:10])
uniques = set(flows)
print(len(uniques), len(pkts[:10]))

# --------------
print('........ Test .........')
sc = StreamCardinalityCounter(W=1)
stream = [(0.1,"A"),(0.11,"A"),(0.2,"A"),(1.3, "A")]
ans = [1,1,1,1]
for s in stream:
    t=s[0]
    sc.add(s[1],t)
    print(f't={t}, items={sc.history}, cardinality: {sc.card(t)}')
print('....... succesful .........\n')

# --------------
print('........ Test .........')
sc = StreamCardinalityCounter(W=1)
stream = [(0.1,"192.168.1.1"),(0.11,"192.168.1.1"),(0.2,"192.168.1.1"),(1.3, "192.168.1.1")]
ans = [1,1,1,1]
for s in stream:
    t=s[0]
    sc.add(s[1],t)
    print(f't={t}, items={sc.history}, cardinality: {sc.card(t)}')
print('....... succesful .........\n')

# --------------
print('........ Test .........')
sc = StreamCardinalityCounter(W=1)
stream = [(0.1,"A"),(0.11,"A"),(0.2,"B"),(1.3, "A")]
ans = [1,1,2,1]
for i,s in enumerate(stream):
    t=s[0]
    sc.add(s[1],t)
    assert sc.card(t) == ans[i]
    print(f't={t}, cardinality: {sc.card(t)}')
print('....... succesful .........\n')

# --------------
print('........ Test .........')
sc = StreamCardinalityCounter(W=1)
stream = [(0.1,"A"),(0.11,"C"),(0.2,"B"),(1.103, "B"),(1.3, "A")]
ans = [1,2,3,2,2]
for i,s in enumerate(stream):
    t=s[0]
    sc.add(s[1],t)
    assert sc.card(t) == ans[i]
    print(f't={t}, cardinality: {sc.card(t)}')
print('....... succesful .........\n')