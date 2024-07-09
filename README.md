# Staggered-HyperLogLog (ST-HLL)

Code for the data structure proposed in the research paper

[Staggered HLL: Near-continuous-time cardinality estimation with no overhead](https://www.sciencedirect.com/science/article/abs/pii/S0140366422002407)  
Computer Communications vol. 193, September 2022  
Authors: *Alessandro Cornacchia, Giuseppe Bianchi, Andrea Bianco, Paolo Giaccone*


## :eagle: Overview
Most existing cardinality estimation algorithms do not natively support interval queries under a sliding window model, making them insensitive to data recency.

**Staggered-HyperLogLog (ST-HLL)** is a probabilistic data structure inspired by HyperLogLog (HLL) that provides nearly continuous-time estimation of cardinality rates, rather than absolute counts. It maintains zero-bit overhead compared to vanilla HLL and introduces negligible additional computational complexity.

## :hamburger: Key-features
- periodic staggered reset of HLL registers
- register equalization at query times to account for counting of different registers over different time spans.
- tested on both synthetic and real Internet traffic traces, ST-HLL is demonstrated to be up to 2x more accurate over the state-of-the-art Sliding HLL, for the same memory demand.

## How to use
We recommend to first familiarize yourself with the basic concepts of the paper prior to using the simulator.

The simulator can read CSV traffic traces and output a cardinality estimate at every packet it processes.

The main parameters are:
- `W`: sliding window size
- `m`: number of HLL registers
 
Run `python hll-sim.py -h` to get the full list of options 
