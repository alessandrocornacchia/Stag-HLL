from itertools import count
import logging
from datetime import datetime
import os
import csv
from bisect import bisect_left
import numpy as np
from include.arrival_rate_functions import *


# configuration for logging   
def configure_logging(loglevel):
    loglevel_id = getattr(logging, loglevel, None)
    if not isinstance(loglevel_id, int):
        raise ValueError('Invalid log level: %s' % loglevel)
    
    format = '%(asctime)s - LOG-LEVEL:%(levelname)s - %(filename)s:%(funcName)s:%(lineno)d - %(message)s'
    logdir = 'results/logs'
    if not os.path.exists(logdir):
        os.mkdir(logdir)    
    file = os.path.join(logdir, datetime.now().strftime("%Y%m%d_%H%M") + '.log')
    logging.basicConfig(filename=None if loglevel=="INFO" else file,
                        level=loglevel_id,
                        format=format,
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    
    logging.info('--------------------------------------------------')
    logging.info('| logging configured succesfully. Log level %s |', loglevel)
    logging.info('--------------------------------------------------')

# strategy pattern for Simulator class: save results to file using this function
def writecsv(dictlist, outfile):
    import pandas as pd
    df = pd.DataFrame(dictlist)
    df.to_csv(outfile+'.csv')

def writepickle(dictlist, outfile):
    import pandas as pd
    df = pd.DataFrame(dictlist)
    df.to_pickle(outfile+'.pkl')

# generate non-homogeneous arrival process
# TODO convert to generator
def generate_arrivals(rate_func, size, deterministic=False, seed=0):
    np.random.seed(seed)
    n = 0
    t = 0
    logging.debug('Generating %d arrival events - lambda(t): %s' % (size, rate_func))
    while n < size:
        (lambda_t, lambda_max) = eval(rate_func)
        if deterministic:
            t = t + 1./lambda_t
        else:
            t = t + np.random.exponential(1./lambda_max)
        if deterministic or np.random.uniform() < lambda_t/lambda_max:
            yield (t, str(n))
            n = n+1

# read trace from file
def read_trace(filename):
    with open(filename, "r") as csvfile:
        datareader = csv.reader(csvfile)
        next(datareader)  # yield the header row
        count = 0
        for row in datareader:
            t = float(row[0])
            flow = '.'.join(row[1:])
            count = count + 1
            yield (t,flow)
    return count

# estimate exact cardinality
# TODO if not unique arrivals np.unique(A[win:i])    
def exact_cardinality(A,i,W):
    tnow = A[i]
    if tnow > W:
        win_start_ptr = bisect_left(A[:i-1], tnow - W)
    else:
        win_start_ptr = 0
    # +1 so to count also border items (oldest and newest inserted)
    return i - win_start_ptr + 1