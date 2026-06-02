# Code for Kidney Allocation Mechanism Comparison

This directory contains all code used to produce the simulation results and
plots in the writeup. The code is organized as one core simulation module
plus three analysis scripts and four plotting scripts.

## Behavioral model

Every mechanism uses the same unified acceptance rule:

```
P(accept) = 0                                            if threshold > theta
          = sigmoid(alpha + beta*(n-1) - delta*k)        if threshold <= theta
```

where `alpha = logit(0.9)`, `beta = 0.10`, `delta in {0, 0.3, 1.0}`, `n` is queue
position, and `k` is the number of prior declines on this kidney. Threshold-violation
auto-declines increment `k` for subsequent patients (the lemons signal does not
distinguish refusal reasons).

## File overview

### Core simulation
**`threshold_sim.py`** — All four mechanisms (omniscient, sequential, batching,
threshold with K-bucket variants) under the unified behavioral model. Exposes
`simulate_run()` which runs one stream of 1500 kidneys for a given mechanism.
`main()` produces the headline results CSV at `threshold_results.csv`.

Mechanism implementations:
- `omniscient`: scan queue, give kidney to first patient with threshold <= theta.
  No logistic, no lemons. Deliberate first-best benchmark.
- `sequential` and `batching`: both call `run_offer_round()` which applies the
  unified rule with batch_size = 1 or 5.
- `threshold`: walks queue, skips non-eligible patients (no time cost), applies
  unified rule to eligible patients. K_cutoffs=None gives continuous threshold.
- `threshold_batched`: hybrid that applies batching within the eligible set.

### Analyses
**`crn_fairness.py`** — Common random numbers analysis. Pre-generates a "world"
(initial queue, kidney quality stream, patient arrival stream, accept-decision
random draws) and runs all four mechanisms against the same world. Per-patient
outcomes feed the Pareto comparison. Outputs `crn_patients.csv`, `crn_positions.csv`,
`crn_pareto.csv`, `crn_position_summary.csv`.

**`marginal_analysis.py`** — Marginal kidney (theta <= 0.15) slice analysis.
Replicates the unified behavioral model from `threshold_sim.py` and records
per-kidney outcomes including recipient threshold and match quality. Outputs
`marginal_kidneys.csv`, `marginal_summary.csv`, `very_marginal_summary.csv`,
`all_kidneys_summary.csv`.

**`fairness_analysis.py`** — Patient outcomes by initial priority tercile and
true threshold tercile. Imports `simulate_run` from `threshold_sim.py`, so it
uses the unified rule directly. Outputs `fairness_priority.csv`,
`fairness_threshold.csv`, `fairness_joint.csv`.

### Plotting (standalone, read the CSVs)
**`plot_threshold.py`** — Pareto frontier, discard vs delta, coarsening cost.
Produces `threshold_pareto.png`, `threshold_discard_vs_delta.png`,
`threshold_coarsening.png`, `threshold_K2_cutoffs.png`.

**`plot_crn.py`** — Patient-level Pareto bars, position-by-quality, position CDF
and histogram. Produces `crn_pareto_bars.png`, `crn_position_by_quality.png`,
`crn_position_cdf.png`, `crn_position_hist.png`.

**`plot_marginal.py`** — Marginal kidney plots. Produces
`marginal_discard_welfare.png`, `marginal_recipient_match.png`,
`coarsening_marginal.png`, `marginal_vs_all.png`.

**`plot_fairness.py`** — Tercile breakdowns. Produces `fairness_priority.png`,
`fairness_priority_lines.png`, `fairness_threshold.png`, `fairness_joint.png`.

## Dependency graph

```
threshold_sim.py  (core; defines world, mechanisms, behavioral model)
    |
    +-- imported by --> fairness_analysis.py
    |
    +-- referenced (logic duplicated) by --> crn_fairness.py
    |                                        marginal_analysis.py
    |
    +-- produces --> threshold_results.csv --> plot_threshold.py

crn_fairness.py --> crn_*.csv --> plot_crn.py
marginal_analysis.py --> marginal_*.csv, all_kidneys_summary.csv --> plot_marginal.py
fairness_analysis.py --> fairness_*.csv --> plot_fairness.py
```

Note: `crn_fairness.py` and `marginal_analysis.py` re-implement the unified rule
locally rather than importing from `threshold_sim.py`, because they also need
patient-level tracking that `simulate_run` doesn't expose. The logic is
identical and was verified by direct code inspection. If you change the
acceptance rule, change it in all three files.

## Reproduction

Run in this order:

```bash
# 1. Core simulation (produces threshold_results.csv, ~5 min)
python threshold_sim.py

# 2. Marginal kidney analysis (~20 min, 10 reps x 7 mechanisms x 3 deltas)
python marginal_analysis.py

# 3. Common random numbers analysis (~10 min, 8 reps x 4 mechanisms x 3 deltas)
python crn_fairness.py

# 4. Fairness tercile analysis (~10 min)
python fairness_analysis.py

# 5. All plots (~1 min total)
python plot_threshold.py
python plot_crn.py
python plot_marginal.py
python plot_fairness.py
```

All scripts write to `/mnt/user-data/outputs/` by default; edit the `OUTPUT_DIR`
constant at the top of each script if running elsewhere.

## Key parameters (in `threshold_sim.py`)

```python
N_KIDNEYS     = 1500   # total kidneys per run
N_WARMUP      = 300    # discarded for steady-state
INITIAL_QUEUE = 150    # initial patients
RHO           = 1.6    # patient arrivals per kidney (matches US ratio)
MAX_HOURS     = 12     # kidney viability
SEED          = 42
N_REPS        = 15     # replications per (mechanism, delta) cell

THRESHOLD_BETA_ALPHA = 2.0   # Beta(2,5): mean threshold ~ 0.29
THRESHOLD_BETA_BETA  = 5.0   # right-skewed, calibrated to ~30% accept marginal
```

The acceptance rule parameters (`alpha_base = logit(0.9)`, `beta_pos = 0.10`)
appear directly in the mechanism code rather than as named constants.

## Computational notes

- All scripts use `numpy.random.default_rng` for reproducibility (seeded by
  `SEED + rep_index * 1000 + ...` to give each cell its own independent stream
  while remaining deterministic).
- Standard errors on discard rates are around 0.001-0.004 with 15 reps;
  differences reported in the writeup are far larger than Monte Carlo noise.
- The simulations are pure NumPy (no JIT, no GPU). Total compute is about
  45 minutes on a modern laptop.
