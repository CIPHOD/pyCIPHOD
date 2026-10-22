# %% Sweeps for OrientEdge vs TPC: 100 edges per s-orientability case (0..3) -> raw + summary CSVs
import itertools
import os
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from pyciphod.utils.scms.dynamic_scm import create_random_linear_dt_dynamic_scm
from pyciphod.utils.time_series.data_format import DTimeVar
from pyciphod.causal_discovery.local.s_orientability import OrientEdge, sOrientability
# NOTE: adjust this import path to wherever run_tpc_bkscg lives
from pyciphod.causal_discovery.basic.tspc_scg import run_tpc_bkscg

NON_ADJACENT = OrientEdge.NON_ADJACENT

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
OUTPUT_DIR = "outputs"
N_REP_PER_CASE = 500
N_JOBS = -1            # 1 = sequential, -1 = all cores
BURN_IN = 200
ORACLE = False
USE_TPC = True
CASES = (0, 1, 2, 3)
CASE_NAMES = {0: "0: non-adjacent", 1: "1: oriented in SCG",
              2: "2: no self-loop", 3: "3: different parents"}
DEFAULT_PARAMS = dict(p=10, p_edge=0.4, gamma_max=2, alpha=0.05, n_timepoints=5000, seed=17102026)
SWEEPS = {
    "p_edge":       [0.2, 0.4, 0.6],
    "p":            [5, 10, 15],
    "n_timepoints": [1000, 5000, 10000],
    "gamma_max":    [1, 2, 3],
    "alpha":        [0.01, 0.05, 0.1],
}
PARAM_KEYS = list(DEFAULT_PARAMS)
METHODS = [("ours", "our"), ("tpc", "tpc")]  # (method name in CSV, column prefix in raw df)


# ----------------------------------------------------------------------------
# Simulation helpers
# ----------------------------------------------------------------------------
def simulate(params, seed):
    """Random SCM -> (scm, ft_dag, scg), deterministic given `seed`."""
    np.random.seed(seed)  # u_dist uses the global RNG
    scm = create_random_linear_dt_dynamic_scm(
        num_ts=params["p"], p_edge=params["p_edge"], causally_stationary=True,
        max_delay=params["gamma_max"], allow_instantaneous=True,
        allow_unmeasured_confounding=False, intercept_low=0, intercept_high=0,
        u_dist=lambda: np.random.normal(0, 1), seed=seed,
    )
    ft_dag = scm[0].induced_ft_dag()
    return scm, ft_dag, ft_dag.get_summary_causal_graph()


def generate_data(scm, scg, params, seed):
    np.random.seed(seed)
    data = scm[0].generate_causally_stationary_time_series_from_latest_mechanisms(
        n_timepoints=params["n_timepoints"], burn_in=BURN_IN, include_latent=False,
        time_series_format="ts_lagged", seed=seed,
    )
    lag_max = scg.get_lag_max()
    return data[[c for c in data.columns if 0 <= -c.time <= lag_max]].copy()


def instantaneous_edges(ft_dag):
    """Set of (u, v) names such that u_t -> v_t in the FT-DAG."""
    t = ft_dag.t_max
    return {(a.get_name(), b.get_name()) for a, b in ft_dag.get_directed_edges()
            if a.get_time() == t and b.get_time() == t}


def pairs_of_case(scg, case):
    """Node pairs (u, v) whose a priori s-orientability case (SCG only) is `case`."""
    sor = sOrientability(scg)
    out = []
    for u, v in itertools.combinations(list(scg.get_vertices()), 2):
        ok, c = sor.is_s_orientable(DTimeVar(u, 0), DTimeVar(v, 0))
        if ok and c == case:
            out.append((u, v))
    return out


# ----------------------------------------------------------------------------
# TPC
# ----------------------------------------------------------------------------
def tpc_edge(tpc, u, v):
    """TPC's verdict on u_t - v_t: 'u -> v', 'u <- v', 'u -- v' (undirected) or 'not adjacent'."""
    A, B = DTimeVar(u, 0), DTimeVar(v, 0)
    if (A, B) in tpc["directed"]:
        return f"{u} -> {v}"
    if (B, A) in tpc["directed"]:
        return f"{u} <- {v}"
    if frozenset((A, B)) in {frozenset(e) for e in tpc["undirected"]}:
        return f"{u} -- {v}"  # undirected: never equal to the truth -> incorrect
    return "not adjacent"


# ----------------------------------------------------------------------------
# Balanced experiment
# ----------------------------------------------------------------------------
def draw_tasks(params):
    """Phase 1 (sequential, cheap): draw SCGs until each case has N_REP_PER_CASE pairs."""
    rng = np.random.default_rng(params["seed"])
    tasks = []
    for case in CASES:
        n_found = attempts = 0
        while n_found < N_REP_PER_CASE and attempts < 500 * N_REP_PER_CASE:
            attempts += 1
            seed = int(rng.integers(0, 2**31 - 1))
            cands = pairs_of_case(simulate(params, seed)[2], case)
            if cands:
                u, v = cands[int(rng.integers(len(cands)))]
                tasks.append(dict(seed=seed, case=case, u=u, v=v, swap=bool(rng.integers(2))))
                n_found += 1
                if n_found % 20 == 0:
                    print(f"    case {case}: {n_found}/{N_REP_PER_CASE} edges drawn "
                          f"({attempts} SCGs)", flush=True)
        if n_found < N_REP_PER_CASE:
            print(f"    /!\\ case {case}: only {n_found}/{N_REP_PER_CASE} edges found")
    return tasks


def evaluate(task, params):
    """Phase 2: run OrientEdge (and TPC) on one pair -> one result row (with timings)."""
    scm, ft_dag, scg = simulate(params, task["seed"])
    data = generate_data(scm, scg, params, task["seed"])
    u, v = task["u"], task["v"]

    # Truth: "cause -> effect" if an instantaneous edge exists, else NON_ADJACENT
    inst = instantaneous_edges(ft_dag)
    cause, effect = (v, u) if (v, u) in inst else (u, v)
    adjacent = (cause, effect) in inst
    truth = f"{DTimeVar(cause, 0)} -> {DTimeVar(effect, 0)}" if adjacent else NON_ADJACENT

    # Ours (output does not depend on argument order, so randomize it); timing covers
    # the orientation only (no simulation / data generation)
    a, b = (v, u) if task["swap"] else (u, v)
    t0 = time.perf_counter()
    oe = OrientEdge(data, scg, alpha=params["alpha"], oracle=ORACLE, true_ftdag=ft_dag)
    pred = oe._run(DTimeVar(a, 0), DTimeVar(b, 0))
    our_time = time.perf_counter() - t0
    row = dict(seed=task["seed"], case=task["case"], u=u, v=v, adjacent=adjacent, truth=truth,
               our_predicted=pred, our_correct=pred == truth, our_undirected=False,
               our_nb_tests=oe.nb_ci_tests, our_time=our_time,
               tpc_predicted=None, tpc_correct=np.nan, tpc_undirected=np.nan,
               tpc_nb_tests=np.nan, tpc_time=np.nan)

    # TPC (compared in its own format, oriented as cause -> effect)
    if USE_TPC:
        # TPC with the SCG BK + support step, run once on the whole dataset
        # (nb_tests and time are those of the whole graph)
        tpc = run_tpc_bkscg(data, scg, alpha=params["alpha"], oracle=ORACLE, true_ftdag=ft_dag)
        tpc_pred = tpc_edge(tpc, cause, effect)
        tpc_truth = f"{cause} -> {effect}" if adjacent else "not adjacent"
        row.update(tpc_predicted=tpc_pred, tpc_correct=tpc_pred == tpc_truth,
                   tpc_undirected="--" in tpc_pred, tpc_nb_tests=tpc["nb_tests"],
                   tpc_time=tpc["time"],
                   tpc_n_support_orientations=tpc["n_support_orientations"],
                   tpc_support_conflicts=len(tpc["support_conflicts"]))
    return row


def run_value(params):
    """All tasks for one parameter setting -> DataFrame (one row per edge)."""
    print("  phase 1: drawing edges", flush=True)
    tasks = draw_tasks(params)
    print(f"  phase 2: evaluating {len(tasks)} edges", flush=True)
    rows = Parallel(n_jobs=N_JOBS, verbose=5)(delayed(evaluate)(t, params) for t in tasks)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
def wilson(k, n, z=1.96):
    """95% Wilson interval for a proportion k/n."""
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    m = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return p, max(0.0, c - m), min(1.0, c + m)


def summarize(df, name, base_params):
    """One row per (param value, case, method): accuracy + Wilson CI, quartiles of #tests and time."""
    num = lambda s: pd.to_numeric(s, errors="coerce").dropna().astype(float)
    rows = []
    for (val, case), g in df.groupby(["param_value", "case"]):
        eff = {**DEFAULT_PARAMS, **base_params, name: val}
        for method, pre in METHODS:
            correct = num(g[f"{pre}_correct"])
            if correct.empty:  # e.g. TPC not run
                continue
            tests, secs = num(g[f"{pre}_nb_tests"]), num(g[f"{pre}_time"])
            k = int(correct.sum())
            p, lo, hi = wilson(k, len(correct))
            q25, med, q75 = np.percentile(tests, [25, 50, 75])
            t25, tmed, t75 = np.percentile(secs, [25, 50, 75])
            rows.append(dict(
                sweep=name, param_value=val, case=case, case_label=CASE_NAMES[case],
                method=method, n=len(correct), n_correct=k,
                correct_prop=p, correct_lo=lo, correct_hi=hi,
                frac_undirected=num(g[f"{pre}_undirected"]).mean(),
                frac_adjacent=g["adjacent"].mean(),
                nb_tests_median=med, nb_tests_q25=q25, nb_tests_q75=q75,
                nb_tests_mean=tests.mean(),
                time_median=tmed, time_q25=t25, time_q75=t75, time_mean=secs.mean(),
                **{k_: eff[k_] for k_ in PARAM_KEYS}))
    return pd.DataFrame(rows)


def print_summary(s):
    """Compact terminal view: accuracy [CI], median #tests and median time (seconds) [q25-q75]."""
    t = pd.DataFrame({
        "param": s["param_value"], "case": s["case"], "method": s["method"], "n": s["n"],
        "accuracy [95% CI]": [f"{r.correct_prop:.2f} [{r.correct_lo:.2f}, {r.correct_hi:.2f}]"
                              for r in s.itertuples()],
        "undir.": [f"{r.frac_undirected:.2f}" for r in s.itertuples()],
        "adj.": [f"{r.frac_adjacent:.2f}" for r in s.itertuples()],
        "CI tests med [q25-q75]": [f"{r.nb_tests_median:.0f} [{r.nb_tests_q25:.0f}-{r.nb_tests_q75:.0f}]"
                                   for r in s.itertuples()],
        "time s med [q25-q75]": [f"{r.time_median:.2f} [{r.time_q25:.2f}-{r.time_q75:.2f}]"
                                 for r in s.itertuples()],
    })
    print("\n" + t.to_string(index=False) + "\n", flush=True)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base_params = dict(DEFAULT_PARAMS)
    summaries = {}  # sweep name -> summary DataFrame (to rebuild the combined file)
    for name, values in SWEEPS.items():
        print(f"\n===== SWEEP {name} = {values} =====", flush=True)
        dfs = []
        for i, val in enumerate(values, 1):
            print(f"[{name}={val}] ({i}/{len(values)})", flush=True)
            t0 = time.time()
            df = run_value({**base_params, name: val})
            df["param_name"], df["param_value"] = name, val
            dfs.append(df)
            print(f"  done in {time.time() - t0:.0f}s", flush=True)
            # checkpoint after each value so a crash never loses finished work
            raw = pd.concat(dfs, ignore_index=True)
            raw.to_csv(f"{OUTPUT_DIR}/sweep_{name}.csv", index=False, na_rep="nan")
            summaries[name] = summarize(raw, name, base_params)
            summaries[name].to_csv(f"{OUTPUT_DIR}/summary_{name}.csv", index=False, na_rep="nan")
            pd.concat(summaries.values(), ignore_index=True).to_csv(
                f"{OUTPUT_DIR}/summary_all_sweeps.csv", index=False, na_rep="nan")
        print_summary(summaries[name])


if __name__ == "__main__":
    main()
# %%