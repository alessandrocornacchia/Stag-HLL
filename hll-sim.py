import argparse
from include.helpers import writepickle, configure_logging
import core
import sys
from include.arrival_rate_functions import LAMBDA_T
from include.builder import hll_algos

def parse_arguments():
    parser = argparse.ArgumentParser(description="Simulator of time-continuous HLL sketch")
    
    # subparsers
    subparsers = parser.add_subparsers(dest='prog')
    subparser_hll = subparsers.add_parser('hll')
    subparser_gt = subparsers.add_parser('ground-truth')

    subparser_hll.add_argument('-m','--num-registers', type=int, nargs='+', help='Number of HLL registers')
    subparser_hll.add_argument('-a','--hll-algorithm', type=str, default='AndreaTimeLogLog', 
                        choices=hll_algos.keys(), help='Cardinality estimation algorithm name')

    prs = [subparser_hll, subparser_gt]
    for p in prs:
        p.add_argument('-q','--query-interval', type=str, help='Query period [s]')
        p.add_argument('-W','--window-duration', type=float, nargs='+', help='Sliding window size in seconds')
        p.add_argument('-o', '--out', type=str, help='Output filename')
        p.add_argument('-L', '--log-level', type=str, choices=['debug', 'info', 'DEBUG', 'INFO'], 
                                    default='info', help='Set log level')
        p.add_argument('-S','--stream-size', type=int, default=100000, help='Number of packets in simulation')
        p.add_argument('-r','--repetitions', default=1, type=int, help='Number of repetitions per experiment with different seeds')
        p.add_argument('-d', '--deterministic-arrivals', action='store_true', default=False,
                                help='Deterministic arrivals at specified rate')
        p.add_argument('-c', '--dump-counters', action='store_true', default=False)
        p.add_argument('-f', '--arrival-rate', type=str, default = LAMBDA_T,
                      help='Arrival rate function used for Poisson arrival process.  \
                      Check \'include/arrival_rate_functions.py\' for a list of available \
                      functions or to define new ones')
        p.add_argument('-i', '--input-trace', type=str, help='Read from this input trace')
        
        
    args = parser.parse_args()
    d = vars(args)
    return d
        
        
''' entry point '''
if __name__ == '__main__':
    args = parse_arguments()
    configure_logging(args['log_level'].upper())
    sim = core.Simulator(args, output_strategy=writepickle)
    sim.start()