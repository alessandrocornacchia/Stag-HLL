import logging
import os
import numpy as np
import time
import simpy
import copy
from include.processes import *
from include.helpers import read_trace
from include.helpers import writecsv
from include.builder import build_hll
from include.pbars import SimTimeBar
from global_ import Runtime
from tqdm import tqdm
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
            #logging.debug(f'[t={t:.5f} s] M: {hll.M}')

    # small utility to evaluate correctly query interval
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
        outf = os.path.normpath(os.path.join(resdir, 'results', fname))
        logging.info(f'Saving to {outf}')
        self.save(results, outf)
    
    # handle case of ground-truth
    def gt_handler(self):
        r = self.args['repetitions']
        W = self.args['window_duration']
        lambda_t = self.args['arrival_rate']
        gaps = self.args['duplicate_spacing']
        results = []
        for ri in range(r):
            # fix simualtion seed
            np.random.seed(ri)
            
            for Wi in W:
                for gap in gaps:   
                    print('')
                    logging.info(f'Algo: ExactCounting, Reps: {ri}, Window {Wi} [s]')
                    
                    # ---- store statistics on current run
                    self.simrun_stats_record['W'] = Wi
                    self.simrun_stats_record['algorithm'] = 'True'
                    self.simrun_stats_record['cardinalities'] = []
                    self.simrun_stats_record['registers_equalized'] = []
                    self.simrun_stats_record['registers'] = []
                    self.simrun_stats_record['nhits'] = []
                    self.simrun_stats_record['hllcards'] = []
                    
                    # create new simpy environment
                    env = simpy.Environment()
                    Runtime.set(env)
                    
                    # create link to send stream through
                    link = simpy.Store(env)
                    
                    # ------ generate traffic or read trace --------
                    if self.args['input_trace'] is None:
                        lambda_t = self.args['arrival_rate']
                        #S = self.args['stream_size']
                        simend = self.args['sim_time']
                        #self.simrun_stats_record['num_items'] = S
                        self.simrun_stats_record['lambda_t'] = lambda_t
                        #self.simrun_stats_record['flow_size'] = fs
                        self.simrun_stats_record['num_cbr'] = self.args['num_cbr']
                        ds = self.__eval_query_interval(gap)
                        self.simrun_stats_record['duplicate_spacing'] = ds
                        self.simrun_stats_record['rep'] = ri
                        
                        # create progress bar
                        pbar = SimTimeBar(simend=simend, 
                                              desc= 'Stream Processing', 
                                              disable=(not self.args['show_progress']))

                        # synthetic stream arrival process
                        env.process(synthetic_traffic(  link,
                                                        ds, 
                                                        self.args['num_cbr'],
                                                        lambda_t,
                                                        self.args['deterministic_arrivals']))
                    else:
                        f = self.args['input_trace']
                        self.simrun_stats_record['lambda_t'] = os.path.split(f)[-1]
                        
                        # open file
                        filesize = os.path.getsize(f)
                        record_size = 40
                        num_pkts = round(filesize/record_size)
                        stream = read_trace(f)

                        # create progress bar
                        pbar = tqdm(total=simend, desc='Stream Processing', disable=(not self.args['show_progress']))
                        
                        # process trace
                        simend = env.process(process_packet_stream(stream, link))
                    
                    # initialize measurement application (hll)
                    sc = StreamCardinalityCounter(Wi)                    
                    env.process(measurement_application(sc, link, pbar))

                    # --- initialize query process  ---
                    Runtime.get().process(self.async_query_process(sc))
                    
                    # run simulator
                    Runtime.get().run(until=simend)
                    
                    # close progress bar once simulation ended
                    pbar.close()

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
        gaps = self.args['duplicate_spacing']
        bits = self.args['timestamp_bits']

        results = []
        for ri in range(r):
            # fix simualtion seed
            np.random.seed(ri)
            
            for Wi in W:
                for mi in m:
                    for b in bits:
                        for gap in gaps:
                            print('')
                            
                            logging.info(f'Algo: {algorithm}, Registers: {mi}, Reps: {ri}, Window: {Wi:.5f} s, cbr packet gap: {gap}')
                            
                            # ---- store statistics on current run
                            self.simrun_stats_record['W'] = Wi
                            self.simrun_stats_record['m'] = mi
                            self.simrun_stats_record['timestamp_bits'] = b
                            self.simrun_stats_record['algorithm'] = algorithm
                            self.simrun_stats_record['cardinalities'] = []
                            self.simrun_stats_record['registers_equalized'] = []
                            self.simrun_stats_record['registers'] = []
                            self.simrun_stats_record['nhits'] = []
                            self.simrun_stats_record['hllcards'] = []
                            
                            # create new simpy environment
                            env = simpy.Environment()
                            Runtime.set(env)
                            
                            # create link to send stream through
                            link = simpy.Store(env)
                            
                            # ------ generate traffic or read trace --------
                            if self.args['input_trace'] is None:
                                lambda_t = self.args['arrival_rate']
                                #S = self.args['stream_size']
                                simend = self.args['sim_time']
                                #self.simrun_stats_record['num_items'] = S
                                self.simrun_stats_record['lambda_t'] = lambda_t
                                #self.simrun_stats_record['flow_size'] = fs
                                self.simrun_stats_record['num_cbr'] = self.args['num_cbr']
                                ds = self.__eval_query_interval(gap)
                                self.simrun_stats_record['duplicate_spacing'] = ds
                                self.simrun_stats_record['rep'] = ri
                                
                                # create progress bar based on simulation time
                                pbar = SimTimeBar(simend=simend, 
                                                desc= 'Stream Processing', 
                                                disable=(not self.args['show_progress']))

                                # synthetic stream arrival process
                                env.process(synthetic_traffic( link, 
                                                                ds, 
                                                                self.args['num_cbr'],
                                                                lambda_t,
                                                                self.args['deterministic_arrivals']))
                            else:
                                f = self.args['input_trace']
                                self.simrun_stats_record['lambda_t'] = os.path.split(f)[-1]
                                
                                # open file
                                filesize = os.path.getsize(f)
                                record_size = 40
                                num_pkts = round(filesize/record_size)
                                stream = read_trace(f)

                                # create progress bar based on total number of packets
                                pbar = tqdm(total=num_pkts, desc='Stream Processing', disable=(not self.args['show_progress']))
                                
                                # process trace
                                simend = env.process(process_packet_stream(stream, link))
                        
                            # initialize measurement process (e.g., hll)
                            hll = build_hll(algorithm, Wi, mi, b)   
                            env.process(measurement_application(hll, link, pbar))
        
                            # --- initialize query process  ---
                            # TODO encapsulate inside hll a function that dumps the registers and call it from outside (in async_query_process)
                            if q:
                                Runtime.get().process(self.async_query_process(hll))
                                if 'StaggeredHLL' in algorithm and dump:
                                    Runtime.get().process(self.dump_reg(hll))   
                            # -- initialize register reset process ---
                            # TODO strategy inside HLL -> in the constructor start a reset process that is however something given from outside (independent
                            # from specific hll implementation)
                            if 'Staggered' in algorithm:
                                Runtime.get().process(self.reset_register(hll))
                            
                            # run simulator
                            Runtime.get().run(until=simend)
                            
                            # close progress bar once simulation ended
                            pbar.close()

                            if algorithm == 'SlidingPCSA':
                                self.simrun_stats_record['renewals'] = hll.Brt

                            # write to file
                            results.append(copy.copy(self.simrun_stats_record))
        return results
            