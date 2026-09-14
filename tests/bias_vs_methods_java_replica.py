#%%
'''
    1:1 replica of the Java SketchAccuracyPlot "sliding kernel" scenario stream generation,
    driving the *unmodified* Python reference sketches (StaggeredHyperLogLog, SlidingHyperLogLog)
    via simpy's Runtime clock exactly like bias_vs_methods.py does.

    Goal: reproduce, in the trusted Python implementation, the exact stream shape and
    parameters used by count-distinct-algorithms' SketchAccuracyPlot.java, so ST-HLL vs
    SlidingHLL behavior can be compared 1:1 against the Java run. No sketch code is touched;
    only the traffic-generation process is adapted to replay the Java-equivalent schedule
    through simpy's clock.

    Stream shape and sketch parameters are read from a YAML config (default:
    tests/configs/java_replica.yaml), with the following structure:

        sketch:
          lg_k: 12                 # p -> m = 2**lg_k registers
          sliding_window: 40000.0  # W

        generator:
          seed: 42
          distribution:
            type: dirichlet
            unique_items: 30000    # size of the item vocabulary
            alpha: 0.9              # Dirichlet concentration (lower = more skewed)
          stages:                   # arbitrary list of arrival-rate stages, run in order
            - name: leading_tail
              events: 40
              scale: 4000.0        # mean inter-arrival time (same time unit as sliding_window)
            ...

    Item identity: categorical draw over unique_items labels ("c_0".."c_{unique_items-1}"),
    weights drawn once from Dirichlet(alpha) via per-item Gamma(alpha,1) draws normalized to
    sum 1 (same construction as EventStreamGenerator's Marsaglia-Tsang/boost Gamma sampler;
    numpy's Generator.dirichlet is mathematically equivalent -- same distribution, different
    RNG stream naturally, since Java's Random and numpy's PCG64 aren't bit-compatible). The
    vocabulary and its weights are drawn once per run and shared by every stage.

    Run:  python tests/bias_vs_methods_java_replica.py [config.yaml] [n_events]
'''
import sys
sys.path.append("..")
sys.path.append(".")

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import simpy

from global_ import Runtime
from algorithms.hll import StaggeredHyperLogLog
from algorithms.shll import SlidingHyperLogLog
from algorithms.exact import StreamCardinalityCounter

DEFAULT_CONFIG_PATH = 'tests/configs/java_replica.yaml'


def load_config(path=DEFAULT_CONFIG_PATH):
    with open(path) as f:
        return yaml.safe_load(f)
# -----------------------------------------------------------------------------------------------


def build_stream_from_config(cfg):
    '''
    Builds the full (timestamp, item) event list from a generator config: one Dirichlet-
    weighted vocabulary shared across all stages, exponential inter-arrivals with a
    per-stage scale override, a single running clock carried across stages (matches
    EventStreamGenerator + the --kernel sliding branch of SketchAccuracyPlot.main()).
    '''
    gen = cfg['generator']
    dist = gen['distribution']
    if dist['type'] != 'dirichlet':
        raise ValueError(f"Unsupported distribution type: {dist['type']!r}")

    rng = np.random.default_rng(gen['seed'])

    # Dirichlet(alpha,...,alpha) weight vector over the vocabulary, drawn once (same role as
    # EventStreamGenerator's constructor-time Gamma draw + normalize).
    unique_items = dist['unique_items']
    labels = np.array([f'c_{i}' for i in range(unique_items)])
    probs = rng.dirichlet(np.full(unique_items, dist['alpha']))

    clock = 0.0
    timestamps = []
    items = []

    def emit_stage(n_events, scale):
        nonlocal clock
        if n_events <= 0:
            return
        # Exponential inter-arrivals + categorical draw, vectorized per stage but still
        # sequential in effect on the shared clock (matches EventStreamGenerator.next()).
        deltas = -scale * np.log(1.0 - rng.random(n_events))
        clock_seq = clock + np.cumsum(deltas)
        clock = clock_seq[-1]
        drawn = rng.choice(labels, size=n_events, p=probs)
        timestamps.extend(clock_seq.tolist())
        items.extend(drawn.tolist())

    for stage in gen['stages']:
        emit_stage(stage['events'], stage['scale'])

    return timestamps, items


def run(cfg, n_events=None):
    '''
    Replays the Java-equivalent stream through simpy's Runtime clock, querying every sketch
    after every event (matching SketchAccuracyPlot's per-event addValue+estimate loop), and
    returns a DataFrame with one row per (event, algo).

    n_events: if set, stop after injecting this many events instead of the full stream
    (temporary knob for debugging termination).
    '''
    timestamps, items = build_stream_from_config(cfg)
    if n_events is not None:
        timestamps = timestamps[:n_events]
        items = items[:n_events]

    sliding_window = cfg['sketch']['sliding_window']
    m = 1 << cfg['sketch']['lg_k']

    env = simpy.Environment()
    Runtime.set(env)

    exact = StreamCardinalityCounter(sliding_window)
    st_hll = StaggeredHyperLogLog(sliding_window, m=m)
    sliding_hll = SlidingHyperLogLog(sliding_window, m=m)

    records = []

    def traffic_and_measure():
        prev_t = 0.0
        for t, item in zip(timestamps, items):
            yield Runtime.get().timeout(t - prev_t)
            prev_t = t

            exact.add(item)
            st_hll.add(item)
            sliding_hll.add(item)

            true = exact.card(t)
            records.append((t, 'ST-HLL', st_hll.card(t), true))
            records.append((t, 'SlidingHLL', sliding_hll.card(t), true))
        # signal completion: reset_process() below runs forever (while True) and would
        # otherwise keep simpy's event queue non-empty indefinitely, so a plain env.run()
        # (no `until`) never returns. Terminate explicitly once the stream is exhausted.
        Runtime.terminate()

    def reset_process():
        while True:
            yield Runtime.get().timeout(st_hll.to)
            st_hll.circular_reset()

    env.process(traffic_and_measure())
    env.process(reset_process())
    env.run(until=Runtime.end_sim())

    df = pd.DataFrame(records, columns=['t', 'algo', 'est', 'true'])
    df['rel_err'] = 100.0 * (df['est'] - df['true']) / df['true'].clip(lower=1e-10)
    return df


def main():
    import time
    config_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG_PATH
    n_events = int(sys.argv[2]) if len(sys.argv) > 2 else None

    cfg = load_config(config_path)
    dist = cfg['generator']['distribution']
    m = 1 << cfg['sketch']['lg_k']

    t0 = time.time()
    df = run(cfg, n_events=n_events)
    print(f'run() took {time.time()-t0:.1f}s for {len(df)//2} events', flush=True)
    df.to_csv('results/bias_vs_methods_java_replica_raw.csv', index=False)

    # ---- error table, same metrics as Java's printErrorTable ----
    print(f'{"Sketch":<15} {"bias":>8} {"mae":>8} {"p50":>8} {"p90":>8} {"p99":>8} {"max":>8}')
    print('-' * 70)
    for algo, sub in df.groupby('algo'):
        abs_err = sub['rel_err'].abs()
        print(f'{algo:<15} {sub["rel_err"].mean():>+7.1f}% {abs_err.mean():>7.1f}% '
              f'{abs_err.quantile(0.50):>7.1f}% {abs_err.quantile(0.90):>7.1f}% '
              f'{abs_err.quantile(0.99):>7.1f}% {abs_err.max():>7.1f}%')

    # ---- plot: estimate vs ground truth, and |relative error| (log scale) ----
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    gt = df[df['algo'] == 'ST-HLL'][['t', 'true']]  # true is identical across algos per t
    axes[0].plot(gt['t'], gt['true'], color='black', lw=2, label='Ground truth')
    for algo, sub in df.groupby('algo'):
        axes[0].plot(sub['t'], sub['est'], lw=1, label=algo)
    axes[0].set_ylabel('Estimate')
    axes[0].legend()
    axes[0].set_title(f"Java-replica stream: Sliding Window (W={cfg['sketch']['sliding_window']:.0f}), "
                       f"unique_items={dist['unique_items']}, alpha={dist['alpha']}, m={m}")

    for algo, sub in df.groupby('algo'):
        axes[1].plot(sub['t'], sub['rel_err'].abs().clip(lower=1e-6), lw=1, label=algo)
    axes[1].set_yscale('log')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('|Relative error| (%, log scale)')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('results/bias_vs_methods_java_replica.png', dpi=150)


if __name__ == '__main__':
    main()
