from global_ import Runtime
from include.helpers import poisson_arrivals
import logging
import numpy as np

'''
    Constant bit rate flow of specified size
'''
def cbr_flow(id, link, size=np.inf, ia=0, delay=0):    
    
    # initial delay
    yield Runtime.get().timeout(delay)
    
    tx = 0
    while tx < size:
        # transmit packet
        t = Runtime.get().now
        logging.debug(f'[t={t:.10f} s] FLOW={id}, SEQ={tx}')
        link.put(id)
        tx += 1
        # wait inter-packet time
        yield Runtime.get().timeout(ia)



''' consume flows produced by traffic stream (here is simple 
    cardinality measurement application) '''
def measurement_application(algorithm, link, pbar=None):    
    while True:
        item = yield link.get()
        # add new element
        algorithm.add(item)
        # update based on delta time elpased
        if pbar:
            pbar.update()


''' Executes stream algorithm '''
def process_packet_stream(stream, link):
    
    while True:
        # read next element from trace (interrupt when finished)
        try:
            ta, item = next(stream)
        except StopIteration:
            break

        # schedule next flow arrival
        t = Runtime.get().now
        dt = ta - t
        yield Runtime.get().timeout(dt)

        # process packet
        logging.debug(f'[t={t:.10f} s] FLOW={item}')
        link.put(item)
        
        # new simpy process that transmit single packet
        #flow_ = cbr_flow(item, link, size=1)
        #Runtime.get().process(flow_)
            
            

''' Simulate synthetic traffic arriving according to Poisson process and
    with given characteristic 
'''
def synthetic_traffic(link, gap, num_cbr, arrival_rate, deterministic):
    
    # create a set of persistent background flows
    for id in range(1,num_cbr+1):
        flow_ = cbr_flow(str(-id), link, ia=gap, delay=gap/2)
        Runtime.get().process(flow_)

    # non-stationary poisson process generating new flows
    arrivals = poisson_arrivals(arrival_rate, deterministic)
    
    while True:
        try:
            t = Runtime.get().now
            # schedule next flow arrival
            ta, item = next(arrivals)
            yield Runtime.get().timeout(ta-t)
            
            # process packet
            logging.debug(f'[t={t:.10f} s] FLOW={item}, SEQ=0')
            link.put(item)
        except StopIteration:
            break
