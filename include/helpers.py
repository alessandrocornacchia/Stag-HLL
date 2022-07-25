from itertools import count
import logging
from datetime import datetime
from global_ import Runtime
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
    file = os.path.join(logdir, datetime.now().strftime("%Y%m%d_%H%M") + '.log')
    create_directory_tree(file)
    logging.basicConfig(filename=None if loglevel=="INFO" else file,
                        level=loglevel_id,
                        format=format,
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    
    logging.info('--------------------------------------------------')
    logging.info('| logging configured succesfully. Log level %s |', loglevel)
    logging.info('--------------------------------------------------')


''' 
    Check if path exists, otherwise creates all directories 
'''
def create_directory_tree(path):
    p = os.path.split(path)
    dir = os.path.join(*p[:-1])
    try:
        os.makedirs(dir)
    except OSError:
        pass
    
# strategy pattern for Simulator class: save results to file using this function
def writecsv(dictlist, outfile):
    import pandas as pd
    create_directory_tree(outfile)
    df = pd.DataFrame(dictlist)
    df.to_csv(outfile+'.csv')

def writepickle(dictlist, outfile):
    import pandas as pd
    create_directory_tree(outfile)
    df = pd.DataFrame(dictlist)
    df.to_pickle(outfile+'.pkl')

''' 
    Generator that read trace from file 
'''
def read_trace(filename):
    with open(filename, "r") as csvfile:
        datareader = csv.reader(csvfile)
        count = 0
        for row in datareader:
            
            try:
                t = float(row[0])
            except:
                continue # skip if header
            
            flow = '.'.join(row[1:])
            count = count + 1
            yield (t, flow)
    return count


'''
    Flow generator according to non-homogeneous arrival process with specified rate
'''
def poisson_arrivals(rate_func, deterministic=False):
    
    n = 0
    t = 0
    logging.debug(f'Starting Poisson process with lambda(t)={rate_func}')
    while True: #n < size:
        
        (lambda_t, lambda_max) = eval(rate_func)
        
        if lambda_max < 0:
            raise ValueError("Arrival rate function must be positive")
        elif lambda_max == 0:
            logging.debug(f'Nothing to generate: Poisson process to be terminated')
            break

        if deterministic:
            t = t + 1./lambda_t
        else:
            t = t + np.random.exponential(1./lambda_max)
        
        if deterministic or np.random.uniform() < lambda_t/lambda_max:
            yield (t, str(n))
            n = n+1
