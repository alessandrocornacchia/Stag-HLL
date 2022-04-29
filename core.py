import logging
import os
import numpy as np
import time
import simpy
import copy
from include.helpers import generate_arrivals
from include.helpers import read_trace
from include.helpers import writecsv
from include.builder import build_hll
from global_ import Runtime
from tqdm import trange, tqdm
from algorithms.exact import StreamCardinalityCounter


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
    def stream_proc(self, stream, algorithm, S, enable_pbar):
        t = 0
        ta, item = next(stream)
        dt = ta-t
        with tqdm(total=S, desc='Stream Processing', disable=(not enable_pbar)) as pbar:
            while True:
                # schedule next arrival
                yield Runtime.get().timeout(dt)
                t = Runtime.get().now
                logging.debug(f'[t={t:.5f} s] arrival of item {item}')
                # add new element
                algorithm.add(item)
                pbar.update()
                # compute next inter-arrival if other elements
                try:
                    ta, item = next(stream)
                except StopIteration:
                    break
                dt = ta - t
                

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
        self.simrun_stats_record['cardinalities'].append( (t, card) )

    # dump statistics about registers
    def dump_reg(self, hll):
        timeout = self.__eval_query_interval(self.args['query_interval'])
        while True:
            yield Runtime.get().timeout(timeout)
            t = Runtime.get().now
            offset = np.floor(t/hll.to).astype(int)
            k = (offset + hll.k) % hll.m
            n = copy.copy(hll.n)
            M = copy.copy(hll.M)
            hllcards = copy.copy(hll.eM)
            self.simrun_stats_record['nhits'].append((t, n[k]))
            self.simrun_stats_record['registers'].append((t,M[k]))
            self.simrun_stats_record['registers_equalized'].append( (t, hllcards[k]))
        

    """
        Perform circular reset of hll register (operations synchronous to register reset can
        be passed as list of callbacks)
    """
    def reset_register(self, hll, callbacks=[]):
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
        if 'W' in _str:
            W = self.simrun_stats_record['W']
        if 'm' in _str:
            m = self.simrun_stats_record['m']
        return eval(_str)

    # simulation entry point
    def start(self):
        if self.args['prog'] == 'ground-truth':
            results = self.gt_handler()
        elif self.args['prog'] == 'hll':
            results  = self.hll_handler()
        
        resdir = os.path.dirname(__file__)
        fname = self.args['out'] if self.args['out'] is not None else time.strftime("%Y%m%d_%H%m%S")
        outf = os.path.join(resdir, 'results', fname)
        logging.info(f'Saving to {outf}')
        self.save(results, outf)
    
    # handle case of ground-truth
    def gt_handler(self):
        r = self.args['repetitions']
        W = self.args['window_duration']
        lambda_t = self.args['arrival_rate']
        pbar = self.args['show_progress']

        results = []
        for ri in range(r):
            # fix simualtion seed
            np.random.seed(ri)
            
            for Wi in W:
                
                # ------ generate arrivals --------
                if self.args['input_trace'] is None:
                    lambda_t = self.args['arrival_rate']
                    S = self.args['stream_size']
                    A = generate_arrivals(lambda_t, S, self.args['deterministic_arrivals'], seed=ri)
                else:
                    f = self.args['input_trace']
                    filesize = os.path.getsize(f)
                    S = round(filesize/40)
                    A = read_trace(f)

                # ---- simulation run statistics
                self.simrun_stats_record['rep'] = ri
                self.simrun_stats_record['num_items'] = S
                self.simrun_stats_record['W'] = Wi
                self.simrun_stats_record['lambda_t'] = lambda_t
                self.simrun_stats_record['algorithm'] = 'True'
                self.simrun_stats_record['cardinalities'] = []

                # ---------- run discrete-event simulator  --------------
                # create new environment for current iteration 
                Runtime.set(simpy.Environment())
                print('')
                logging.info(f'Algo: ExactCounting, Reps: {ri}, Window {Wi} [s]')
                sc = StreamCardinalityCounter(Wi)
                # -- initialize stream arrival process --
                stream_proc = Runtime.get().process(self.stream_proc(A, sc, S, pbar))
                # --- initialize query process  ---
                Runtime.get().process(self.async_query_process(sc))
                # -- run simulation
                Runtime.get().run(until=stream_proc)                    
                # ----------------------------------------------------------
                results.append(copy.copy(self.simrun_stats_record))
        return results

    # handle case we want to simulate HLL
    def hll_handler(self):
        # short notation
        r = self.args['repetitions']
        W = self.args['window_duration']
        m = self.args['num_registers']        
        
        q = self.args['query_interval'] is not None
        algorithm = self.args['hll_algorithm']
        dump = self.args['dump_counters']
        pbar = self.args['show_progress']

        results = []
        for ri in range(r):
            # fix simualtion seed
            np.random.seed(ri)
            
            for Wi in W:
                for mi in m:

                    print('')
                    hll = build_hll(algorithm, Wi, mi)   
                    
                    # ------ generate arrivals --------
                    if self.args['input_trace'] is None:
                        lambda_t = self.args['arrival_rate']
                        S = self.args['stream_size']
                        self.simrun_stats_record['num_items'] = S
                        self.simrun_stats_record['lambda_t'] = lambda_t
                        A = generate_arrivals(lambda_t, S, self.args['deterministic_arrivals'], seed=ri)
                    else:
                        f = self.args['input_trace']
                        self.simrun_stats_record['lambda_t'] = os.path.split(f)[-1]
                        filesize = os.path.getsize(f)
                        row_bytes = 40
                        S = round(filesize/row_bytes)
                        A = read_trace(f)

                    # ---- simulation run statistics
                    self.simrun_stats_record['rep'] = ri
                    self.simrun_stats_record['W'] = Wi
                    self.simrun_stats_record['m'] = mi
                    self.simrun_stats_record['cardinalities'] = []
                    self.simrun_stats_record['registers_equalized'] = []
                    self.simrun_stats_record['registers'] = []
                    self.simrun_stats_record['nhits'] = []
                    self.simrun_stats_record['algorithm'] = str(hll)
                    self.simrun_stats_record['hllcards'] = []
                    
                    # ---------- run discrete-event simulator  --------------
                    # create new environment for current iteration 
                    Runtime.set(simpy.Environment())
                    logging.info(f'Algo: {hll}, Registers: {mi}, Reps: {ri}, Window: {Wi:.5f} s')
                    # -- initialize stream arrival process --
                    stream_proc = Runtime.get().process(self.stream_proc(A, hll, S, pbar))
                    # --- initialize query process  ---
                    if q:
                        Runtime.get().process(self.async_query_process(hll))
                        if algorithm == 'Stag-HLL' and dump:
                            Runtime.get().process(self.dump_reg(hll))
                    # -- initialize register reset process ---
                    if algorithm == 'Stag-HLL':
                        Runtime.get().process(self.reset_register(hll))
                    
                    Runtime.get().run(until=stream_proc)
                    # ----------------------------------------------------------
                    results.append(copy.copy(self.simrun_stats_record))
        return results
            