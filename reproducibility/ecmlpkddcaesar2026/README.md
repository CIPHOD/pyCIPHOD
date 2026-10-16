# Root cause analysis via difference graph discovery - from linear time-series data - ECMLPKDD  Caesar

This folder contains the evaluation script used to benchmark the temporal difference-graph algorithms and the baselines on synthetic time-series data generated.

The main entry point is:

```bash
experiments.py
```

The script generates normal and anomalous regimes, runs the selected algorithms, scores their predictions against the known changed mechanisms, and writes summary tables with the mean, standard deviation, and variance of the F1 score.

---

## Expected project layout

Current development layout:

```text
RBAL/
├── generator.py
│── experiments.py
│── baseline/
│    ├── MBGH.py
│    ├── microcause.py
│    ├── estimation.py
│    ├── rcd.py
│    └── utils_rcd.py
│    └── tspc_compare.py
```

---

## What `experiments.py` does

For each selected setting, number of nodes, sample size, and repetition, the script:

1. Generates one synthetic run using `generate_one_run` from `generator.py`.
2. Extracts the ground-truth.
3. Runs the selected algorithms.
4. Scores graph algorithms.
---

## Available algorithms

The active algorithms are:

```text
tsldiffpc      Temporal linear difference PC
tsldiffpc_pc   tsLDiffPC with additional tPC orientation
tsdci          Temporal DCI
tsdci_pc       tsDCI with additional tPC orientation
tsMBGH         MBGH baseline
microcause     MicroCause root-cause baseline
rcd            RCD root-cause baseline
tPCUnion       tPC in each regime and union of the two graphs
```

`tsiSCAN` is intentionally disabled for now.


---

## Results

To reproduce the results of the paper run :

```bash
python ./experiments.py \
  --settings setting2_lag1 \
  --p-list 3 \
  --n-list 1000 \
  --n-reps 10 \
  --user-lags 1 \
  --change-model single_edge all_parents all_parents_min2 \
  --algos tsldiffpc tsldiffpc_pc tsdci tsdci_pc tsMBGH microcause rcd tspc_compare
```

