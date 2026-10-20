# %% Sweeps: our method (OrientInstantaneousEdge) vs TPC  ->  summary CSVs for the paper tables
"""
Runs the balanced-by-case sweeps and writes, in OUTPUT_DIR:

  sweep_<param>.csv          raw results, one row per evaluated edge
  summary_<param>.csv        one row per (param value, case, method): everything needed
                             to build the table of that sweep
  summary_all_sweeps.csv     the four summary_<param>.csv stacked in one file

Files are rewritten after EACH value of a sweep, so a crash or a Ctrl-C never loses
the finished work.

Columns of the summary CSVs
---------------------------
  sweep, param_value          swept parameter and its value
  case, case_label            0..3 and its short name
  method                      "ours" or "tpc"
  n                           number of edges in the cell
  n_correct                   number of correctly oriented edges
  correct_prop                proportion of correct edges            (panel (a), point value)
  correct_lo, correct_hi      95% Wilson interval                    (panel (a), interval)
  frac_undirected             fraction of edges left undirected (counted as incorrect)
  nb_tests_median             median number of CI tests              (panel (b), point value)
  nb_tests_q25, nb_tests_q75  quartiles                              (panel (b), interval)
  nb_tests_mean               mean number of CI tests
  p, p_edge, gamma_max,       effective simulation parameters of the cell
  n_timepoints, alpha, seed

Table rule used in the paper: bold = better method of the cell, i.e. the larger
`correct_prop` in panel (a) and the smaller `nb_tests_median` in panel (b).
"""

import itertools
import os
import warnings

import numpy as np
import pandas as pd

try:  # optional: parallel evaluation
    from joblib import Parallel, delayed
except ImportError:  # pragma: no cover
    Parallel = None

from pyciphod.utils.scms.dynamic_scm import create_random_linear_dt_dynamic_scm
from pyciphod.utils.time_series.data_format import DTimeVar
from pyciphod.causal_discovery.local.orientation_using_scg import (
    OrientInstantaneousEdge,
    sOrientability,
    SCGIncompatibilityWarning,
)
from pyciphod.utils.graphs.background_knowledge import BackgroundKnowledge
from pyciphod.causal_discovery.basic.ts_constraint_based import TsPC
from pyciphod.utils.stat_tests.independence_tests import FisherZTest
from pyciphod.utils.graphs.partially_specified_graphs import SummaryCausalGraph

# ----------------------------------------------------------------------------
# Global configuration
# ----------------------------------------------------------------------------
SAVE_OUTPUTS = True      # write CSVs to OUTPUT_DIR
OUTPUT_DIR = "outputs"
BURN_IN = 200
ORACLE = True
N_JOBS = -1              # >1: evaluate edges in parallel (needs joblib); -1: all cores

# seed=None -> not reproducible; seed=<int> -> fully reproducible
DEFAULT_PARAMS = dict(p=10, p_edge=0.4, gamma_max=2, alpha=0.05, n_timepoints=10000, seed=None)

CASE_NAMES = {
    0: "0: non-adjacent",
    1: "1: oriented in SCG",
    2: "2: no self-loop",
    3: "3: different parents",
}
PARAM_KEYS = ["p", "p_edge", "gamma_max", "n_timepoints", "alpha", "seed"]
METHODS = [("ours", "our"), ("tpc", "tpc")]  # (method name in the CSV, column prefix in the raw df)


# ----------------------------------------------------------------------------
# Small helpers (saving, parallelism)
# ----------------------------------------------------------------------------

def _out(name):
    """Path inside OUTPUT_DIR (directory created on demand)."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return os.path.join(OUTPUT_DIR, name)


def maybe_to_csv(df, name, verbose=True):
    if SAVE_OUTPUTS:
        path = _out(name)
        df.to_csv(path, index=False, na_rep="nan")
        if verbose:
            print(f"  -> saved to {path}")


def _pmap(fn, items, n_jobs=None):
    """Map `fn` over `items`, in parallel if requested and joblib is available."""
    n_jobs = N_JOBS if n_jobs is None else n_jobs
    if n_jobs == 1 or Parallel is None or len(items) <= 1:
        return [fn(i) for i in items]
    return Parallel(n_jobs=n_jobs)(delayed(fn)(i) for i in items)


# ----------------------------------------------------------------------------
# Graph / SCM utils
# ----------------------------------------------------------------------------

def bk_from_scg(scg: SummaryCausalGraph, gamma_max: int) -> BackgroundKnowledge:
    """Background knowledge derived from the SCG (forbidden edges + mandatory orientations)."""
    bk = BackgroundKnowledge()
    nodes = list(scg.get_vertices())
    all_kl = list(itertools.product(range(gamma_max + 1), repeat=2))

    for x, y in itertools.combinations(nodes, 2):
        if y not in set(scg.get_adjacencies(x)):
            for k, l in all_kl:
                bk.add_forbidden_edge(DTimeVar(x, -k), DTimeVar(y, -l))
                bk.add_forbidden_edge(DTimeVar(y, -k), DTimeVar(x, -l))
        else:
            for a, b in ((x, y), (y, x)):  # a -> b only (not a <-> b)
                if a in set(scg.get_parents(b)) and a not in set(scg.get_children(b)):
                    bk.add_mandatory_orientation(DTimeVar(a, 0), DTimeVar(b, 0))

    for x in nodes:
        if x not in set(scg.get_adjacencies(x)):  # no self-loop -> no lagged copies of x
            for k, l in all_kl:
                if k != l:
                    bk.add_forbidden_edge(DTimeVar(x, -k), DTimeVar(x, -l))
    return bk


def true_edge(ft_dag_edges, u, v):
    u2v = (DTimeVar(u, 0), DTimeVar(v, 0)) in ft_dag_edges
    v2u = (DTimeVar(v, 0), DTimeVar(u, 0)) in ft_dag_edges
    if u2v and v2u:
        return f"{u}t <-> {v}t"  # degenerate
    if u2v:
        return f"{u}t -> {v}t"
    if v2u:
        return f"{u}t <- {v}t"
    return "not adjacent"


def tpc_edge(directed, undirected, u, v):
    A, B = DTimeVar(u, 0), DTimeVar(v, 0)
    if (A, B) in directed:
        return f"{u}t -> {v}t"
    if (B, A) in directed:
        return f"{u}t <- {v}t"
    if frozenset((A, B)) in {frozenset(e) for e in undirected}:
        return f"{u}t -- {v}t"  # undirected: never equal to the truth -> counted as incorrect
    return "not adjacent"


def _simulate(params, seed):
    """Random SCM -> (scm, ft_dag, scg). Deterministic given `seed`."""
    np.random.seed(seed)  # u_dist uses the global RNG
    p, pe, g = params["p"], params["p_edge"], params["gamma_max"]
    scm = create_random_linear_dt_dynamic_scm(
        num_ts=p, p_edge=pe, causally_stationary=True, max_delay=g,
        allow_instantaneous=True, allow_unmeasured_confounding=False,
        weight_low=-1 / (p * pe * g), weight_high=1 / (p * pe * g),
        intercept_low=0.0, intercept_high=0.0,
        u_dist=lambda: np.random.normal(0, 1), seed=seed,
    )
    ft_dag = scm[0].induced_ft_dag()
    return scm, ft_dag, ft_dag.get_summary_causal_graph()


def _generate_data(scm, scg, params, seed):
    np.random.seed(seed)
    lag_max = scg.get_lag_max()
    data = scm[0].generate_causally_stationary_time_series_from_latest_mechanisms(
        n_timepoints=params["n_timepoints"], burn_in=BURN_IN, include_latent=False,
        time_series_format="ts_lagged", seed=seed,
    )
    return data[[c for c in data.columns if 0 <= -c.time <= lag_max]].copy()


def _classify(scg, u, v):
    """sOrientability.classify() for the pair (u_t, v_t): 0, 1, 2, 3 or None (not s-orientable)."""
    return sOrientability(scg, DTimeVar(u, 0), DTimeVar(v, 0)).classify()


def find_edges_by_case(scg):
    """{case: [(u, v), ...]} for all node pairs, using sOrientability.classify()."""
    by_case = {0: [], 1: [], 2: [], 3: []}
    for u, v in itertools.combinations(list(scg.get_vertices()), 2):
        c = _classify(scg, u, v)
        if c in by_case:
            by_case[c].append((u, v))
    return by_case


# ----------------------------------------------------------------------------
# Running the methods
# ----------------------------------------------------------------------------

def run_tpc(data, scg, ft_dag, params):
    """Run TPC once on the whole dataset (nb_tests counts the CI tests of the whole graph)."""
    tpc = TsPC(
        sparsity=params["alpha"], ci_test=FisherZTest,
        background_knowledge=bk_from_scg(scg, scg.get_lag_max()),
        oracle=ORACLE, true_ftdag=ft_dag,
    )
    tpc.run(data)
    return dict(directed=set(tpc.g_hat.get_directed_edges()),
                undirected=set(tpc.g_hat.get_undirected_edges()),
                nb_tests=tpc.nb_ci_tests)


def evaluate_edge(u, v, params, scg, ft_dag, data, tpc, extra):
    """Evaluate our method (and TPC, if `tpc` is not None) on the edge u-v -> one result row.

    An edge left undirected (by us: 'ut - vt', by TPC: 'ut -- vt') never matches the truth,
    hence is counted as incorrect for both methods."""
    truth = true_edge(set(ft_dag.get_directed_edges()), u, v)

    orienter = OrientInstantaneousEdge(
        data=data, A=DTimeVar(u, 0), B=DTimeVar(v, 0), scg=scg,
        alpha=params["alpha"], ci_test=FisherZTest, oracle=ORACLE, true_ftdag=ft_dag,
    )
    with warnings.catch_warnings():  # the fallback warning is expected in experiments
        warnings.simplefilter("ignore", SCGIncompatibilityWarning)
        our_pred, _ = orienter.orient()

    tpc_pred = tpc_edge(tpc["directed"], tpc["undirected"], u, v) if tpc is not None else None
    return dict(
        **extra, u=u, v=v, truth=truth,
        our_predicted=our_pred, our_correct=our_pred == truth,
        our_undirected=our_pred == orienter.undirected,
        our_nb_tests=orienter.nb_ci_tests,
        tpc_predicted=tpc_pred,
        tpc_correct=(tpc_pred == truth) if tpc is not None else np.nan,
        tpc_undirected=("--" in tpc_pred) if tpc is not None else np.nan,
        tpc_nb_tests=tpc["nb_tests"] if tpc is not None else np.nan,
    )


# ----------------------------------------------------------------------------
# Balanced-by-case experiment
# ----------------------------------------------------------------------------

def _draw_balanced_tasks(n_rep_per_case, params, cases, max_attempts_per_case):
    """Phase 1 (sequential, cheap): draw random SCGs until each case has enough edges.
    Returns a list of tasks (seed, case, u, v). Reproducible if params['seed'] is set."""
    rng = np.random.default_rng(params["seed"])
    max_attempts = max_attempts_per_case or n_rep_per_case * 500  # safeguard
    tasks = []
    for case in cases:
        collected = attempts = 0
        while collected < n_rep_per_case and attempts < max_attempts:
            attempts += 1
            seed = int(rng.integers(0, 2**31 - 1))
            _, _, scg = _simulate(params, seed)
            candidates = find_edges_by_case(scg)[case]
            if not candidates:
                continue
            u, v = candidates[int(rng.integers(0, len(candidates)))]
            tasks.append(dict(seed=seed, case=case, u=u, v=v))
            collected += 1
        if collected < n_rep_per_case:
            print(f"  /!\\ Case {case}: only {collected}/{n_rep_per_case} edges found "
                  f"after {attempts} attempts.")
    return tasks


def run_balanced_experiment(n_rep_per_case, params=None, cases=(0, 1, 2, 3), use_tpc=True,
                            max_attempts_per_case=None):
    """
    For each case: draw random SCGs until one contains an edge of that case, pick one such edge
    at random and evaluate the methods on it, until `n_rep_per_case` edges are collected.
    Phase 1 draws the tasks sequentially; phase 2 evaluates them (in parallel if N_JOBS != 1).
    """
    params = {**DEFAULT_PARAMS, **(params or {})}
    tasks = _draw_balanced_tasks(n_rep_per_case, params, cases, max_attempts_per_case)

    def evaluate(item):
        idx, t = item
        scm, ft_dag, scg = _simulate(params, t["seed"])
        data = _generate_data(scm, scg, params, t["seed"])
        tpc = run_tpc(data, scg, ft_dag, params) if use_tpc else None
        return evaluate_edge(
            t["u"], t["v"], params, scg, ft_dag, data, tpc,
            dict(experiment_id=idx, seed=t["seed"], target_case=t["case"]),
        )

    return pd.DataFrame(_pmap(evaluate, list(enumerate(tasks))))


def run_sweep(param_name, values, base_params, n_rep_per_case=100, use_tpc=False,
              on_value_done=None):
    """Balanced experiment for each value of `param_name` -> long DataFrame (1 row per edge).

    `on_value_done(df_so_far)` is called after each value (used to checkpoint the CSVs)."""
    dfs = []
    for val in values:
        print(f"[sweep {param_name}={val}]")
        df = run_balanced_experiment(n_rep_per_case, {**base_params, param_name: val}, use_tpc=use_tpc)
        df["param_name"], df["param_value"] = param_name, val
        dfs.append(df)
        if on_value_done is not None:
            on_value_done(pd.concat(dfs, ignore_index=True))
    return pd.concat(dfs, ignore_index=True)


# ----------------------------------------------------------------------------
# Summary statistics (everything needed to build the tables)
# ----------------------------------------------------------------------------

def _wilson_ci(data, z=1.96):
    """(proportion, low, high) with Wilson interval, for 0/1 data."""
    n = len(data)
    p = np.mean(data)
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    lo, hi = max(0.0, center - margin), min(1.0, center + margin)
    return min(max(p, lo), hi), lo, hi  # clip p against floating-point errors


def summarize_sweep(df, param_name, base_params):
    """Summary of one sweep: one row per (param value, case, method).

    `base_params` are the sweep's fixed parameters; the effective parameters of each cell
    ({**DEFAULT_PARAMS, **base_params, param_name: value}) are stored in the p, p_edge, ... columns."""
    rows = []
    for val in sorted(df["param_value"].unique()):
        eff = {**DEFAULT_PARAMS, **base_params, param_name: val}
        for case in sorted(int(c) for c in df["target_case"].dropna().unique()):
            sub = df[(df["param_value"] == val) & (df["target_case"] == case)]
            for method, prefix in METHODS:
                if f"{prefix}_correct" not in sub.columns:
                    continue
                correct = sub[f"{prefix}_correct"].dropna().astype(float).values
                if len(correct) == 0:  # e.g. TPC not run
                    continue
                tests = sub[f"{prefix}_nb_tests"].dropna().astype(float).values
                und_col = f"{prefix}_undirected"
                frac_und = (float(sub[und_col].dropna().astype(float).mean())
                            if und_col in sub.columns and sub[und_col].notna().any() else np.nan)
                p, lo, hi = _wilson_ci(correct)
                q25, q75 = np.percentile(tests, [25, 75])
                rows.append(dict(
                    sweep=param_name, param_value=val,
                    case=case, case_label=CASE_NAMES.get(case, str(case)), method=method,
                    n=len(correct), n_correct=int(correct.sum()),
                    correct_prop=p, correct_lo=lo, correct_hi=hi,
                    frac_undirected=frac_und,
                    nb_tests_median=float(np.median(tests)), nb_tests_q25=float(q25),
                    nb_tests_q75=float(q75), nb_tests_mean=float(np.mean(tests)),
                    **{k: eff[k] for k in PARAM_KEYS},
                ))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main_sweep(n_rep_per_case=3, use_tpc=True):
    base_params = dict(p=3, p_edge=0.4, gamma_max=1, alpha=0.05, n_timepoints=5000, seed=123)
    sweeps = {
        "p_edge": [0.4],
        "p": [5],
        "n_timepoints": [5000],
        "gamma_max": [1],
    }
    summaries = {}  # sweep name -> summary DataFrame (kept to rebuild the combined file)

    def checkpoint(name, df, verbose):
        """Rewrite raw CSV + summary CSV of this sweep + combined summary CSV."""
        maybe_to_csv(df, f"sweep_{name}.csv", verbose)
        summaries[name] = summarize_sweep(df, name, base_params)
        maybe_to_csv(summaries[name], f"summary_{name}.csv", verbose)
        maybe_to_csv(pd.concat(summaries.values(), ignore_index=True),
                     "summary_all_sweeps.csv", verbose)

    for name, values in sweeps.items():
        df = run_sweep(name, values, base_params, n_rep_per_case=n_rep_per_case, use_tpc=use_tpc,
                       on_value_done=lambda d, name=name: checkpoint(name, d, verbose=False))
        checkpoint(name, df, verbose=True)  # final save (prints the paths)


if __name__ == "__main__":
    main_sweep()

# %%