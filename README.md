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

## Java-port replica test

`tests/bias_vs_methods_java_replica.py` reproduces, in this trusted Python codebase, the exact
synthetic stream shape and sketch parameters used by `count-distinct-algorithms`'
`SketchAccuracyPlot.java` (`--kernel sliding`), so `StaggeredHyperLogLog` and `SlidingHyperLogLog`
can be compared 1:1 against a Java run. No sketch code is touched, only the traffic-generation
process.

The stream is built from a sequence of stages (e.g. a slow leading tail, a ramp-up, a dense
plateau, a trailing tail), each with its own event count and arrival-rate parameter, drawing
item identities from a single Dirichlet-weighted vocabulary shared across all stages.

Setup:

```bash
uv sync
```

Run (uses `tests/configs/java_replica.yaml` by default):

```bash
uv run python tests/bias_vs_methods_java_replica.py
```

Run with a custom config, and/or capped at a fixed number of events (useful for quickly
checking a config before committing to a full run):

```bash
uv run python tests/bias_vs_methods_java_replica.py path/to/config.yaml [n_events]
```

Outputs `results/bias_vs_methods_java_replica_raw.csv` (per-event, per-sketch estimate vs
ground truth) and `results/bias_vs_methods_java_replica.png` (estimate and relative-error
plots over time).

### Config format

```yaml
sketch:
  lg_k: 12                 # p -> m = 2**lg_k registers
  sliding_window: 40000.0  # W, same time unit as each stage's `scale` below

generator:
  seed: 42
  distribution:
    type: dirichlet
    unique_items: 30000     # size of the item vocabulary
    alpha: 0.9               # Dirichlet concentration (lower = more skewed)
  stages:                    # arbitrary list of arrival-rate stages, run in order
    - name: leading_tail
      events: 40
      scale: 4000.0         # mean inter-arrival time (exponential), same unit as sliding_window
    - name: ramp
      events: 35000
      scale: 2.0
    - name: plateau
      events: 16500
      scale: 2.0
    - name: trailing_tail
      events: 40
      scale: 4000.0
```

- `sketch.lg_k` / `sketch.sliding_window`: passed directly to `StaggeredHyperLogLog` and
  `SlidingHyperLogLog`.
- `generator.seed`: seeds the single RNG stream used for the whole generation process (vocabulary
  weights, every stage's inter-arrival draws, every stage's item draws) — one seed reproduces
  the entire stream.
- `generator.distribution`: item-population model for the vocabulary. Currently only
  `type: dirichlet` is supported.
- `generator.stages`: run in the order listed; each stage draws `events` items with exponential
  inter-arrival times of mean `scale`, on a single running clock carried across stages.
