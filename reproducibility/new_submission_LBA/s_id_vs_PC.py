# %% Sweeps for OrientEdge-based adjustment sets (ConstructAdjSet) vs TPC (SCG BK) vs oracle FT-DAG
# 100 effects per query (ATE, CDE) and per parameter value -> raw + summary CSVs (no figures).
# Same parameters, sweeps and export layout as the s-orientability sweeps script.
import os
# Avoid BLAS oversubscription under joblib (otherwise timings are meaningless)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from pyciphod.utils.scms.dynamic_scm import create_random_linear_dt_dynamic_scm
from pyciphod.utils.time_series.data_format import DTimeVar, wide_timevar_to_lagged_df
# NOTE: adjust this import path to wherever sIdentifiability / ConstructAdjSet live
from pyciphod.causal_discovery.local.s_identifiability import sIdentifiability, ConstructAdjSet
# NOTE: adjust this import path to wherever run_tpc_bkscg lives
from pyciphod.causal_discovery.basic.tspc_scg import run_tpc_bkscg

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
OUTPUT_DIR = "outputs"          # files are prefixed "adjset_" so they never clash with the orientability ones
N_REP_PER_QUERY = 500           # effects DRAWN per query and per parameter value (skipped tasks are not replaced)
N_JOBS = -1                     # 1 = sequential, -1 = all cores
BURN_IN = 200
ORACLE = False
USE_TPC = True
QUERIES = ("ATE", "CDE")
MAX_EFFECT_LAG = None
MIN_ABS_EFFECT = 0.1
MAX_SPECTRAL_RADIUS = 0.95      # reject (nearly) explosive SCMs: OLS/CI tests are meaningless there
DEFAULT_PARAMS = dict(p=10, p_edge=0.4, gamma_max=2, alpha=0.05, n_timepoints=5000, seed=17102026)
SWEEPS = {
    "p_edge":       [0.2, 0.4, 0.6],
    "p":            [5, 10, 15],
    "n_timepoints": [1000, 5000, 10000],
    "gamma_max":    [1, 2, 3],
    "alpha":        [0.01, 0.05, 0.1],
}
PARAM_KEYS = list(DEFAULT_PARAMS)
METHODS = [("ours", "our"), ("tpc", "tpc"), ("ftdag", "ftdag")]  # (method name in CSV, column prefix in raw df)


class TpcUndirectedNeighbor(Exception):
    """TPC left an undirected edge incident to the target."""
    def __init__(self, msg, edges=()):
        super().__init__(msg)
        self.edges = list(edges)


class SkipTask(Exception):
    """Task excluded for ALL methods (keeps the comparison paired)."""
    def __init__(self, reason, detail=""):
        super().__init__(reason)
        self.reason, self.detail = reason, detail


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


def generate_data(scm, params, seed, window):
    np.random.seed(seed)
    wide = scm[0].generate_causally_stationary_data_from_latest_mechanisms(
        n_timepoints=params["n_timepoints"], n_samples=1, burn_in=BURN_IN,
        include_latent=False, seed=seed, reindex_time=True,
    )
    lagged = wide_timevar_to_lagged_df(wide, window_size=window)
    return lagged[[c for c in lagged.columns if 0 <= -c.time <= window - 1]].copy()


# ----------------------------------------------------------------------------
# Ground truth
# ----------------------------------------------------------------------------
def coefficient_template(scm):
    model, coefs, _ = scm
    observed = set(model.v)
    latest = {}
    for n in observed:
        latest[n.name] = max(latest.get(n.name, n.time), n.time)
    return {(s.name, t.name, t.time - s.time): w
            for (s, t), w in coefs.items()
            if s in observed and t in observed and t.time == latest[t.name]}


def _coef_matrices(template, names):
    n = len(names)
    idx = {name: i for i, name in enumerate(names)}
    p = max((lag for (_, _, lag) in template), default=0)
    B = [np.zeros((n, n)) for _ in range(p + 1)]  # B[k][tgt, src]
    for (s, t, lag), w in template.items():
        B[lag][idx[t], idx[s]] = w
    return B, idx, p


def impulse_responses(template, names, h_max):
    """Psi[h][y, x] = effect of X_t on Y_{t+h} (sum over directed paths of coefficient products)."""
    n = len(names)
    B, idx, p = _coef_matrices(template, names)
    inv = np.linalg.inv(np.eye(n) - B[0])
    psi = [inv]
    for h in range(1, h_max + 1):
        acc = sum((B[k] @ psi[h - k] for k in range(1, min(h, p) + 1)), np.zeros((n, n)))
        psi.append(inv @ acc)
    return psi, idx


def spectral_radius(template, names):
    """Spectral radius of the companion matrix of the reduced-form VAR (<1 <=> stationary)."""
    n = len(names)
    B, _, p = _coef_matrices(template, names)
    if p == 0:
        return 0.0
    inv = np.linalg.inv(np.eye(n) - B[0])
    C = np.zeros((n * p, n * p))
    for k in range(1, p + 1):
        C[:n, (k - 1) * n:k * n] = inv @ B[k]
    if p > 1:
        C[n:, :-n] = np.eye(n * (p - 1))
    return float(np.max(np.abs(np.linalg.eigvals(C))))


def candidate_effects(scm, scg):
    """Only effects whose TARGET series (treatment for ATE, outcome for CDE) has N(S_T, SCG) = {}."""
    gamma = scg.get_lag_max()
    cap = gamma if MAX_EFFECT_LAG is None else min(MAX_EFFECT_LAG, gamma)
    sid = sIdentifiability(scg)
    ok = {n: not sid._non_s_orientable_with(n) for n in scg.get_vertices()}

    template = coefficient_template(scm)
    names = sorted({n.name for n in scm[0].v})
    out = {"ATE": [], "CDE": []}
    if spectral_radius(template, names) >= MAX_SPECTRAL_RADIUS:
        return out  # unstable SCM: discarded
    psi, idx = impulse_responses(template, names, cap)

    for (s, t, lag), w in template.items():
        if lag <= cap and ok.get(t, False) and abs(w) >= MIN_ABS_EFFECT:
            out["CDE"].append((s, t, lag, float(w)))
    for nx in names:
        if not ok.get(nx, False):
            continue
        for ny in names:
            for h in range(cap + 1):
                if nx == ny and h == 0:
                    continue
                val = float(psi[h][idx[ny], idx[nx]])
                if abs(val) >= MIN_ABS_EFFECT:
                    out["ATE"].append((nx, ny, h, val))
    return out


# ----------------------------------------------------------------------------
# Estimation
# ----------------------------------------------------------------------------
def estimate_effect(data, X, Y, adj):
    cols = [X] + sorted(set(adj) - {X, Y})
    missing = [c for c in cols + [Y] if c not in data.columns]
    if missing:
        raise KeyError(f"columns missing from data (window too small?): {missing}")
    sub = data[cols + [Y]].dropna()
    A = np.column_stack([np.ones(len(sub)), sub[cols].to_numpy(float)])
    coef = np.linalg.lstsq(A, sub[Y].to_numpy(float), rcond=None)[0]
    return float(coef[1]), len(cols) - 1


# ----------------------------------------------------------------------------
# TPC
# ----------------------------------------------------------------------------
def tpc_target_parents(tpc, nT):
    T0 = DTimeVar(nT, 0)
    bad = [e for e in tpc["undirected"] if T0 in e]
    if bad:
        raise TpcUndirectedNeighbor(f"undirected edge(s) incident to {T0}", bad)
    return {A for (A, B) in tpc["directed"] if B == T0}


def diagnose_undirected(edges, nT, scg):
    """For each undirected TPC edge at the target, say whether its other endpoint's series is
    (wrongly) in N(S_T, SCG). If every one is 'ok', TPC is just weaker than the FT-MPDAG;
    any 'NON_S_ORIENTABLE' means the task selection (s-orientability check) is inconsistent."""
    nso = sIdentifiability(scg)._non_s_orientable_with(nT)
    T0 = DTimeVar(nT, 0)
    parts = []
    for e in edges:
        for o in [x for x in e if x != T0]:
            tag = "NON_S_ORIENTABLE" if o.get_name() in nso else "ok"
            parts.append(f"{o}[lag={-o.get_time()}]:{tag}")
    return "; ".join(parts)


# ----------------------------------------------------------------------------
# Experiment
# ----------------------------------------------------------------------------
def draw_tasks(params):
    """Phase 1 (sequential, cheap): draw SCMs until each query has N_REP_PER_QUERY effects."""
    rng = np.random.default_rng(params["seed"])
    tasks = []
    for query in QUERIES:
        n_found = attempts = 0
        while n_found < N_REP_PER_QUERY and attempts < 500 * N_REP_PER_QUERY:
            attempts += 1
            seed = int(rng.integers(0, 2**31 - 1))
            scm, _, scg = simulate(params, seed)
            cands = candidate_effects(scm, scg)[query]
            if cands:
                nX, nY, lag, truth = cands[int(rng.integers(len(cands)))]
                tasks.append(dict(seed=seed, query=query, nX=nX, nY=nY, lag=lag, truth=truth))
                n_found += 1
                if n_found % 20 == 0:
                    print(f"    {query}: {n_found}/{N_REP_PER_QUERY} effects drawn "
                          f"({attempts} SCMs)", flush=True)
        if n_found < N_REP_PER_QUERY:
            print(f"    /!\\ {query}: only {n_found}/{N_REP_PER_QUERY} effects found")
    return tasks


def _metrics(beta, truth):
    if beta is None or not np.isfinite(beta):
        return dict(beta=np.nan, abs_err=np.nan, rel_err=np.nan)
    return dict(beta=beta, abs_err=abs(beta - truth), rel_err=abs(beta - truth) / abs(truth))


def _fill(row, pre, status, beta, truth, adj_size, nb_tests, secs):
    row.update({f"{pre}_status": status, f"{pre}_adj_size": adj_size,
                f"{pre}_nb_tests": nb_tests, f"{pre}_time": secs,
                **{f"{pre}_{k}": v for k, v in _metrics(beta, truth).items()}})


def _evaluate(task, params):
    scm, ft_dag, scg = simulate(params, task["seed"])
    gamma = scg.get_lag_max()
    query, nX, nY, lag, truth = (task[k] for k in ("query", "nX", "nY", "lag", "truth"))
    is_te = query == "ATE"
    X, Y = DTimeVar(nX, -lag), DTimeVar(nY, 0)
    nT, u = (nX, -lag) if is_te else (nY, 0)

    data = generate_data(scm, params, task["seed"], window=lag + gamma + 1)
    row = dict(seed=task["seed"], query=query, nX=nX, nY=nY, lag=lag, truth=truth,
               target=nT, error=None, skip_reason=None, skip_detail=None)

    # Oracle FT-DAG: adjustment = true parents of the target, 0 CI tests
    t_max = ft_dag.t_max
    T_node = DTimeVar(nT, t_max)
    fparents = {DTimeVar(a.get_name(), a.get_time() - t_max + u)
                for a, b in ft_dag.get_directed_edges() if b == T_node}
    row["ftdag_outcome_in_set"] = Y in fparents
    beta, k = estimate_effect(data, X, Y, fparents)
    _fill(row, "ftdag", "ok", beta, truth, k, 0, 0.0)

    # Ours (timing covers the adjustment-set construction only)
    t0 = time.perf_counter()
    cas = ConstructAdjSet(data, scg, alpha=params["alpha"], oracle=ORACLE, true_ftdag=ft_dag)
    res = cas.run(query, X, Y)
    our_time = time.perf_counter() - t0
    if res == ConstructAdjSet.NOT_S_ORIENTABLE:
        # should never happen when N(S_T) = {}: flag it loudly, skip the task
        raise SkipTask("ours_not_s_orientable")
    if res == ConstructAdjSet.ZERO:
        _fill(row, "our", "zero", 0.0, truth, 0, cas.nb_ci_tests, our_time)
    else:
        beta, k = estimate_effect(data, X, Y, res)
        _fill(row, "our", "ok", beta, truth, k, cas.nb_ci_tests, our_time)

    # TPC with the SCG BK + support step, run once on the whole dataset
    # (a failure skips the whole task: nothing is added for any method)
    if USE_TPC:
        tpc = run_tpc_bkscg(data, scg, alpha=params["alpha"], oracle=ORACLE, true_ftdag=ft_dag)
        try:
            parents = tpc_target_parents(tpc, nT)
        except TpcUndirectedNeighbor as e:
            detail = diagnose_undirected(e.edges, nT, scg)
            raise SkipTask("tpc_undirected_neighbor",
                           f"{detail} | support_conflicts={tpc['support_conflicts']}")
        adj = {DTimeVar(A.get_name(), A.get_time() + u) for A in parents}
        row["tpc_outcome_in_set"] = Y in adj
        row["tpc_n_support_orientations"] = tpc["n_support_orientations"]
        beta, k = estimate_effect(data, X, Y, adj)
        _fill(row, "tpc", "ok", beta, truth, k, tpc["nb_tests"], tpc["time"])
    return row


def evaluate(task, params):
    """Phase 2: one effect -> one result row. Never let one failing task kill the whole run."""
    base = dict(seed=task["seed"], query=task["query"], nX=task["nX"], nY=task["nY"],
                lag=task["lag"], truth=task["truth"], error=None,
                skip_reason=None, skip_detail=None)
    try:
        return _evaluate(task, params)
    except SkipTask as s:
        return {**base, "skip_reason": s.reason, "skip_detail": s.detail}
    except Exception as e:  # noqa: BLE001
        return {**base, "error": f"{type(e).__name__}: {e}"}


def run_value(params):
    """All tasks for one parameter setting -> DataFrame (one row per effect)."""
    print("  phase 1: drawing effects", flush=True)
    tasks = draw_tasks(params)
    print(f"  phase 2: evaluating {len(tasks)} effects", flush=True)
    rows = Parallel(n_jobs=N_JOBS, verbose=5)(delayed(evaluate)(t, params) for t in tasks)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
def _clean(df):
    """Tasks usable for the comparison: no crash, not skipped (same tasks for every method)."""
    df = df.copy()
    for c in ("error", "skip_reason"):
        if c not in df:
            df[c] = None
    return df[df["error"].isna() & df["skip_reason"].isna()]


def _boot_rrmse_ci(rel, n_boot=2000, seed=0):
    sq = np.asarray(rel, float) ** 2
    if len(sq) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    m = rng.choice(sq, size=(n_boot, len(sq)), replace=True).mean(axis=1)
    return tuple(np.sqrt(np.percentile(m, [2.5, 97.5])))


def summarize(df, name, base_params):
    """
    One row per (param value, query, method), on the tasks where every method produced an estimate.
      rrmse (+ 95% bootstrap CI) : sqrt(mean((beta_hat - truth)^2 / truth^2)), the relative RMSE
      rel_err_*                  : median and quartiles of |beta_hat - truth| / |truth|
      nb_tests_*, time_*         : median, quartiles and mean of #CI tests and seconds
      n_drawn / n_skipped_* / n_crashed : task accounting for the (param value, query) group
    """
    num = lambda s: pd.to_numeric(s, errors="coerce").dropna().astype(float)
    rows = []
    for (val, query), g in df.groupby(["param_value", "query"]):
        eff = {**DEFAULT_PARAMS, **base_params, name: val}
        used = _clean(g)
        skip = g["skip_reason"] if "skip_reason" in g else pd.Series(dtype=object)
        err = g["error"] if "error" in g else pd.Series(dtype=object)
        for method, pre in METHODS:
            if f"{pre}_rel_err" not in used:
                continue
            rel = num(used[f"{pre}_rel_err"])
            if rel.empty:  # e.g. TPC not run
                continue
            tests, secs = num(used[f"{pre}_nb_tests"]), num(used[f"{pre}_time"])
            lo, hi = _boot_rrmse_ci(rel.to_numpy())
            r25, rmed, r75 = np.percentile(rel, [25, 50, 75])
            q25, med, q75 = np.percentile(tests, [25, 50, 75])
            t25, tmed, t75 = np.percentile(secs, [25, 50, 75])
            rows.append(dict(
                sweep=name, param_value=val, query=query, method=method,
                n=len(rel), n_drawn=len(g),
                n_skipped_tpc_undirected=int((skip == "tpc_undirected_neighbor").sum()),
                n_skipped_ours_not_s_orientable=int((skip == "ours_not_s_orientable").sum()),
                n_crashed=int(err.notna().sum()),
                rrmse=float(np.sqrt((rel ** 2).mean())), rrmse_lo=lo, rrmse_hi=hi,
                rel_err_median=rmed, rel_err_q25=r25, rel_err_q75=r75,
                adj_size_median=num(used[f"{pre}_adj_size"]).median(),
                nb_tests_median=med, nb_tests_q25=q25, nb_tests_q75=q75, nb_tests_mean=tests.mean(),
                time_median=tmed, time_q25=t25, time_q75=t75, time_mean=secs.mean(),
                **{k_: eff[k_] for k_ in PARAM_KEYS}))
    return pd.DataFrame(rows)


def print_summary(s):
    """Compact terminal view: relative RMSE [CI], median relative error, median #tests and time (seconds) [q25-q75]."""
    t = pd.DataFrame({
        "param": s["param_value"], "query": s["query"], "method": s["method"],
        "n/drawn": [f"{r.n}/{r.n_drawn}" for r in s.itertuples()],
        "rRMSE [95% CI]": [f"{r.rrmse:.3f} [{r.rrmse_lo:.3f}, {r.rrmse_hi:.3f}]" for r in s.itertuples()],
        "med rel.err": [f"{r.rel_err_median:.3f}" for r in s.itertuples()],
        "CI tests med [q25-q75]": [f"{r.nb_tests_median:.0f} [{r.nb_tests_q25:.0f}-{r.nb_tests_q75:.0f}]"
                                   for r in s.itertuples()],
        "time s med [q25-q75]": [f"{r.time_median:.3f} [{r.time_q25:.3f}-{r.time_q75:.3f}]"
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
            if df["error"].notna().any():
                print(f"  /!\\ {int(df['error'].notna().sum())} task(s) crashed, "
                      f"e.g. {df['error'].dropna().iloc[0]}", flush=True)
            # checkpoint after each value so a crash never loses finished work
            raw = pd.concat(dfs, ignore_index=True)
            raw.to_csv(f"{OUTPUT_DIR}/adjset_sweep_{name}.csv", index=False, na_rep="nan")
            summaries[name] = summarize(raw, name, base_params)
            summaries[name].to_csv(f"{OUTPUT_DIR}/adjset_summary_{name}.csv", index=False, na_rep="nan")
            pd.concat(summaries.values(), ignore_index=True).to_csv(
                f"{OUTPUT_DIR}/adjset_summary_all_sweeps.csv", index=False, na_rep="nan")
        print_summary(summaries[name])


if __name__ == "__main__":
    main()
# %%