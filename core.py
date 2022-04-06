import logging
import os
import numpy as np
import time
import simpy
import copy
from include.helpers import generate_arrivals, exact_cardinality, configure_logging
from include.helpers import writecsv
from include.hyperloglogs.builder import build_hll
from global_ import Runtime
from tqdm import trange, tqdm

'''
    Simulator : gets initialized with all the simulation parameters

    - output_strategy : can customize how to store simulation data. D
                        default is csv with one row corresponding to one simulation run
    
    Simulator(..).start() to launch the simulation. Do not support multithreding at the moment
'''
class Simulator():

    def __init__(self, args, output_strategy = writecsv) -> None:
        self.simrun_stats_record = {}
        self.args = args
        self.save = output_strategy

    # executes stream i.e. sequence of items
    def process_stream(self, hll):
        S = self.simrun_stats_record['num_items']
        A = self.simrun_stats_record['arrival_t']
        
        dt = A[0]
        for i in tqdm(range(S), desc='Stream processing'):
            # schedule next arrival
            yield Runtime.get().timeout(dt)
            t = Runtime.get().now
            logging.debug(f'[t={t:.5f} s] arrival of item {i}')
            # add new element
            hll.add(repr(i))
            # compute next inter-arrival
            if i < S-1:
                dt = A[i+1] - A[i]

    """
        Perform cardinality queries at specified periodicity (asynchronously)
    """
    def async_query_process(self, hll):
        timeout = self.__eval_query_interval(self.args['query_interval'])
        logging.debug(f'Starting query process, sampling interval {timeout}')
        while True:
            yield Runtime.get().timeout(timeout)
            self.query(hll)

    """ query and store result """
    def query(self, hll):
        t = Runtime.get().now
        card = len(hll)
        self.simrun_stats_record['hllcard'].append( (t, card) )

    # dump statistics about registers
    def dump_reg(self, hll):
        t = Runtime.get().now
        offset = np.floor(t/hll.to).astype(int)
        k = (offset + hll.k) % hll.m
        n = copy.copy(hll.n)
        M = copy.copy(hll.M)
        hllcards = copy.copy(hll.x)
        self.simrun_stats_record['reg_i_stats'].append((t,n,M))
        self.simrun_stats_record['reg_it_stats'].append((t,n[k],M[k]))
        self.simrun_stats_record['hllcards'].append( (t, hllcards[k]))
        

    """
        Perform circular reset of hll register (operations synchronous to register reset can
        be passed as list of callbacks)
    """
    def reset_register(self, hll, callbacks):
        while True:
            yield Runtime.get().timeout(hll.to)
            
            # call list of registered callbacks
            for c in callbacks:
                c(hll)

            hll.circular_reset()

            t = Runtime.get().now
            logging.debug(f'[t={t:.5f} s] M: {hll.M}')

    # small utility to evaluate correctly query interval (TODO move outside main simulator class)        
    def __eval_query_interval(self, _str):
        W = self.simrun_stats_record['W']
        m = self.simrun_stats_record['m']
        return eval(_str)

    def start(self):
        # short notation
        r = self.args['repetitions']
        m = self.args['num_registers']        
        S = self.args['stream_size']
        W = self.args['window_duration']
        q = self.args['query_interval'] is not None
        lambda_t = self.args['arrival_rate']
        algorithm = self.args['hll_algorithm']

        results = []
        for ri in range(r):
            # fix simualtion seed
            np.random.seed(ri)
            
            # generate arrivals (either deterministic or stochastic)
            arrival_times = generate_arrivals(lambda_t, S, self.args['deterministic_arrivals'])

            for Wi in W:

                # evaluate ground truth
                x = np.zeros(S)
                print('')
                logging.info(f'Algo: ExactCounting, Reps: {ri}, Window {Wi} [s]')
                for item in trange(S, desc='Stream processing'):
                    x[item] = exact_cardinality(arrival_times, item, Wi)

                for mi in m:
                       
                    # create HLL data structure
                    hll = build_hll(algorithm, Wi, mi)
                    
                    # ---- simulation run statistics
                    self.simrun_stats_record['rep'] = ri
                    self.simrun_stats_record['num_items'] = S
                    self.simrun_stats_record['arrival_t'] = arrival_times
                    self.simrun_stats_record['W'] = Wi
                    self.simrun_stats_record['m'] = mi
                    self.simrun_stats_record['cardinality'] = x
                    self.simrun_stats_record['lambda_t'] = lambda_t
                    self.simrun_stats_record['algorithm'] = str(hll)
                    self.simrun_stats_record['hllcard'] = []
                    self.simrun_stats_record['reg_it_stats'] = []
                    self.simrun_stats_record['reg_i_stats'] = []
                    self.simrun_stats_record['hllcards'] = []
        
                    # ----
                
                    # ---------- run discrete-event simulator  --------------
                    print('')
                    logging.info(f'Algo: {hll}, Registers: {mi}, Reps: {ri}, Window: {Wi:.5f} s')
                    # create new environment for current iteration 
                    Runtime.set(simpy.Environment())
                    # -- initialize stream arrival process --
                    stream_proc = Runtime.get().process(self.process_stream(hll))
                    # --- initialize query process  ---
                    callb = []
                    if q:
                        Runtime.get().process(self.async_query_process(hll))
                    else:
                        callb = [self.query, self.dump_reg]
                    # -- initialize register reset process ---
                    if algorithm == 'Stag-HLL':
                        Runtime.get().process(self.reset_register(hll, callb))
                    Runtime.get().run(until=stream_proc)
                    # ----------------------------------------------------------
                    results.append(copy.copy(self.simrun_stats_record))
                    
            outf = os.path.join('./results', 
                self.args['out'] if self.args['out'] is not None else time.strftime("%Y%m%d_%H%m%S"))
            self.save(results, outf)