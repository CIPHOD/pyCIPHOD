# %% Effet causal estimé : s-identifiabilité (notre méthode) vs TPC vs ensemble d'ajustement du DAG
#
# Pour chaque requête (ATE ou CDE) on compare trois estimateurs OLS qui ne diffèrent que par
# l'ensemble d'ajustement :  (i) DAG (référence)  (ii) sIdentifiability  (iii) TPC.
# Erreur d'un estimateur = (estimation - estimation basée sur le DAG)^2.
# Le rôle des "cas" de l'expérience d'orientation est joué ici par le type de requête (ATE / CDE).

import itertools
import numpy as np
import pandas as pd

from pyciphod.utils.time_series.data_format import DTimeVar
from pyciphod.causal_discovery.local.orientation_using_scg import sIdentifiability
from pyciphod.causal_discovery.basic.ts_constraint_based import TsPC
from pyciphod.utils.stat_tests.independence_tests import FisherZTest

# Utilitaires partagés avec l'expérience de s-orientabilité (même dossier)
from reproducibility.new_submission_LBA.experimentation_vs_PC_old import (
    DEFAULT_PARAMS, _simulate, bk_from_scg, maybe_to_csv, value_stats,
    _num, _ms, _print_table, plot_sweep_comparison,
)

BURN_IN = 200
ORACLE = False          # True : d-séparation exacte ; False : tests CI sur les données
CI = FisherZTest
QUERIES = ("ATE", "CDE")


# ----------------------------------------------------------------------------
# Utilitaires
# ----------------------------------------------------------------------------

def shift(nodes, dt):
    return {DTimeVar(n.get_name(), n.get_time() + dt) for n in nodes}


def parents_dag(ft_dag, gmax, name):
    """Parents réels de name_t (dans le FT-DAG, le temps courant est gmax)."""
    node = DTimeVar(name, gmax)
    return {DTimeVar(u.get_name(), u.get_time() - gmax)
            for u, w in ft_dag.get_directed_edges() if w == node}


def parents_tpc(g_hat, name):
    """Parents orientés de name_t dans le graphe estimé (temps courant = 0)."""
    node = DTimeVar(name, 0)
    return {u for u, w in g_hat.get_directed_edges() if w == node}


def ols_effect(data, outcome, treatment, controls):
    cols = [treatment] + sorted(controls, key=lambda c: (c.get_name(), c.get_time()))
    X = np.column_stack([data[c].to_numpy(dtype=float) for c in cols] + [np.ones(len(data))])
    y = data[outcome].to_numpy(dtype=float)
    # Séries explosives / NaN : on retire les lignes non finies (LAPACK plante sinon)
    keep = np.isfinite(X).all(axis=1) & np.isfinite(y)
    if keep.sum() <= X.shape[1] or keep.mean() < 0.9:   # trop peu de données exploitables
        return np.nan
    try:
        return np.linalg.lstsq(X[keep], y[keep], rcond=None)[0][0]
    except np.linalg.LinAlgError:
        return np.nan


def draw_query(rng, ft_dag, scg, gmax, kind):
    """Tire une requête (T, y) de type `kind`, ou None si le SCG n'en contient pas."""
    nodes = list(scg.get_vertices())
    if kind == "ATE":       # Y descendant de X, T = X_{t-γ}
        cands = [(x, y) for x, y in itertools.permutations(nodes, 2) if y in scg.get_descendants(x)]
        if not cands:
            return None
        x, y = cands[rng.integers(len(cands))]
        return DTimeVar(x, -int(rng.integers(gmax + 1))), y
    # CDE : arête X_{t-γ} -> Y_t du FT-DAG avec X != Y
    cands = [(T, y) for y in nodes for T in parents_dag(ft_dag, gmax, y) if T.get_name() != y]
    return cands[rng.integers(len(cands))] if cands else None


def run_tpc(data, scg, ft_dag, gmax, params):
    """Une seule découverte TPC par DAG, sur la fenêtre [-gmax, 0]."""
    try:
        tpc = TsPC(params["alpha"], ci_test=CI, oracle=ORACLE, true_ftdag=ft_dag,
                   background_knowledge=bk_from_scg(scg, gmax))
        tpc.run(data[[c for c in data.columns if -c.time <= gmax]])
        return dict(g_hat=tpc.g_hat, nb_tests=tpc.nb_ci_tests, error=False)
    except Exception:
        return dict(g_hat=None, nb_tests=np.nan, error=True)


# ----------------------------------------------------------------------------
# Évaluation d'une requête
# ----------------------------------------------------------------------------

def evaluate_query(kind, T, y, ft_dag, scg, gmax, data, tpc, params, extra):
    Y = DTimeVar(y, 0)
    # ATE : Pa(X_{t-γ}) ; CDE : Pa(Y_t)
    name, dt = (T.get_name(), T.get_time()) if kind == "ATE" else (y, 0)

    adj_dag = shift(parents_dag(ft_dag, gmax, name), dt)
    adj_tpc = None if tpc["error"] else shift(parents_tpc(tpc["g_hat"], name), dt)
    try:
        sid = sIdentifiability(data, T, Y, kind, scg, params["alpha"], CI, ORACLE, ft_dag)
        adj_sid, tests_sid, sid_error = sid.get_adj_set(), sid.nb_ci_tests, False
    except Exception:
        adj_sid, tests_sid, sid_error = None, np.nan, True

    est = dict(dag=np.nan, sid=np.nan, tpc=np.nan)
    sets = dict(dag=adj_dag, sid=adj_sid, tpc=adj_tpc)
    if all(s is not None for s in sets.values()):
        sets = {k: s - {T, Y} for k, s in sets.items()}
        if all(c in data.columns for s in sets.values() for c in s):
            est = {k: ols_effect(data, Y, T, s) for k, s in sets.items()}

    return dict(
        **extra, query=kind, T=str(T), Y=str(Y),
        identified=adj_sid is not None and not sid_error,
        sid_error=sid_error, tpc_error=tpc["error"],
        est_dag=est["dag"], est_sid=est["sid"], est_tpc=est["tpc"],
        comparable=not np.isnan(est["dag"] + est["sid"] + est["tpc"]),
        se_sid=(est["sid"] - est["dag"]) ** 2, se_tpc=(est["tpc"] - est["dag"]) ** 2,
        tests_sid=tests_sid, tests_tpc=tpc["nb_tests"],
    )


def run_dag(seed, kinds, params, rng):
    """Un DAG, un run TPC, une requête par type de `kinds` (si elle existe) -> liste de lignes."""
    np.random.seed(seed)
    scm, ft_dag, scg = _simulate(params, seed)
    gmax = scg.get_lag_max()

    queries = [(k, *q) for k in kinds if (q := draw_query(rng, ft_dag, scg, gmax, k))]
    if not queries:
        return []

    full = scm.generate_causally_stationary_time_series_from_latest_mechanisms(
        n_timepoints=params["n_timepoints"], burn_in=BURN_IN, include_latent=False,
        time_series_format="ts_lagged", seed=seed,
    )
    # 2*gmax lags : l'ensemble d'ajustement de l'ATE remonte jusqu'à t-γ-gmax
    data = full[[c for c in full.columns if -c.time <= 2 * gmax]].copy()
    tpc = run_tpc(data, scg, ft_dag, gmax, params)
    return [evaluate_query(k, T, y, ft_dag, scg, gmax, data, tpc, params, dict(seed=seed))
            for k, T, y in queries]


# ----------------------------------------------------------------------------
# Expériences : "globale" (une requête de chaque type par DAG) et équilibrée par type
# ----------------------------------------------------------------------------

def run_experiment(n_sim, params=None, first_seed=2020, verbose=True):
    params = {**DEFAULT_PARAMS, **(params or {})}
    rows = []
    for seed in range(first_seed, first_seed + n_sim):
        rows += run_dag(seed, QUERIES, params, np.random.default_rng(seed))
        if verbose:
            print(f"sim {seed} done")
    return pd.DataFrame(rows)


def run_balanced_experiment(n_rep_per_query, params=None, queries=QUERIES,
                            max_attempts_per_query=None, verbose=True):
    """Pour chaque type de requête : tire des DAGs jusqu'à obtenir `n_rep_per_query` requêtes."""
    params = {**DEFAULT_PARAMS, **(params or {})}
    rng = np.random.default_rng(params["seed"])
    max_attempts = max_attempts_per_query or n_rep_per_query * 500

    rows = []
    for kind in queries:
        collected = attempts = 0
        if verbose:
            print(f"\n=== Searching {kind} queries ===")
        while collected < n_rep_per_query and attempts < max_attempts:
            attempts += 1
            found = run_dag(int(rng.integers(0, 2**31 - 1)), [kind], params, rng)
            rows += found
            collected += len(found)
        if collected < n_rep_per_query:
            print(f"  /!\\ {kind} : only {collected}/{n_rep_per_query} queries after {attempts} attempts.")
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Résumés et tableaux
# ----------------------------------------------------------------------------

def summarize_by_query(df):
    """Une ligne par type de requête + une ligne 'ALL'."""
    rows = []
    for q, g in [*df.groupby("query"), ("ALL", df)]:
        ok = g[g["comparable"]]
        t_sid = value_stats(g["tests_sid"])
        t_tpc = value_stats(g.groupby("seed")["tests_tpc"].first())  # TPC : un run par DAG
        rows.append(dict(
            query=q, n=len(g), n_sid_error=int(g["sid_error"].sum()), n_tpc_error=int(g["tpc_error"].sum()),
            prop_identified=g["identified"].mean(), n_comparable=len(ok),
            mse_sid=ok["se_sid"].mean(), mse_tpc=ok["se_tpc"].mean(),
            med_se_sid=ok["se_sid"].median(), med_se_tpc=ok["se_tpc"].median(),
            tests_sid_mean=t_sid["mean"], tests_sid_std=t_sid["std"],
            tests_tpc_mean=t_tpc["mean"], tests_tpc_std=t_tpc["std"],
        ))
    return pd.DataFrame(rows)


def print_query_table(df_by_query, title="Performance by query type"):
    print(f"\n=== {title} ===")
    rows = [[r["query"], str(int(r["n"])),
             f"{_num(100 * r['prop_identified'], 1)}%",
             f"{int(r['n_comparable'])}",
             f"{_num(r['mse_sid'], 5)} / {_num(r['mse_tpc'], 5)}",
             f"{_num(r['med_se_sid'], 5)} / {_num(r['med_se_tpc'], 5)}",
             _ms(r["tests_sid_mean"], r["tests_sid_std"], 1),
             _ms(r["tests_tpc_mean"], r["tests_tpc_std"], 1)]
            for _, r in df_by_query.iterrows()]
    _print_table(["Query", "N", "s-identified", "N comparable", "MSE (ours / TPC)",
                  "Median SE (ours / TPC)", "CI tests (ours)", "CI tests per DAG (TPC)"], rows)
    print("\n(SE = squared difference to the DAG-based estimate; comparable = all 3 estimates available."
          " CI tests: mean [std].)")


# ----------------------------------------------------------------------------
# Sweep (un paramètre à la fois, équilibré par type de requête)
# ----------------------------------------------------------------------------

def run_sweep(param_name, values, base_params, n_rep_per_query=100):
    dfs = []
    for val in values:
        print(f"[sweep {param_name}={val}]")
        df = run_balanced_experiment(n_rep_per_query, {**base_params, param_name: val}, verbose=False)
        df = df[~df["sid_error"]].copy()
        df["param_name"], df["param_value"] = param_name, val
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def plot_sweep(df, param_name, series, ylabel, title, mode, out_path):
    """Réutilise plot_sweep_comparison : le type de requête tient lieu de 'target_case'."""
    plot_sweep_comparison(df.assign(target_case=df["query"]), param_name, series,
                          ylabel, title, mode=mode, out_path=out_path)


# ----------------------------------------------------------------------------
# Main (choisir avec EXPERIMENT)
# ----------------------------------------------------------------------------

def main_global():
    params = dict(p=5, p_edge=0.4, gamma_max=1, alpha=0.05, n_timepoints=10000)
    df = run_experiment(n_sim=100, params=params)
    maybe_to_csv(df, "identifiability_detail_global.csv")
    by_query = summarize_by_query(df)
    maybe_to_csv(by_query, "identifiability_by_query_global.csv")
    print_query_table(by_query, "Global experiment (one ATE and one CDE query per DAG)")


def main_balanced():
    params = dict(p=5, p_edge=0.4, gamma_max=1, alpha=0.05, n_timepoints=5000, seed=42)
    df = run_balanced_experiment(n_rep_per_query=20, params=params)
    maybe_to_csv(df, "identifiability_detail_balanced.csv")
    by_query = summarize_by_query(df)
    maybe_to_csv(by_query, "identifiability_by_query_balanced.csv")
    print_query_table(by_query, "Balanced sampling (identical N per query type)")


def main_sweep():
    base_params = dict(p=5, p_edge=0.4, gamma_max=1, alpha=0.05, n_timepoints=5000, seed=123)
    sweeps = {
        "p_edge": [0.2, 0.4, 0.6, 0.8],
        "p": [3, 5, 8, 10],
        "n_timepoints": [500, 1000, 5000, 10000, 20000],
        "gamma_max": [1, 2, 3, 4],
    }
    ours = dict(method_label="Our method", method_key="ours")
    tpc = dict(method_label="TPC", method_key="tpc")

    for name, values in sweeps.items():
        df = run_sweep(name, values, base_params, n_rep_per_query=100)
        maybe_to_csv(df, f"sweep_identifiability_{name}.csv")

        # Proportion de requêtes s-identifiées (notre méthode)
        plot_sweep(df, name, [dict(value_col="identified", **ours)],
                   "Proportion s-identified (per query)", "s-identified proportion by query type",
                   "binomial", f"sweep_identifiability_{name}_identified.png")

        # Erreur d'estimation (vs DAG), notre méthode vs TPC, sur les cas comparables
        plot_sweep(df, name, [dict(value_col="se_sid", **ours), dict(value_col="se_tpc", **tpc)],
                   "Squared error vs DAG-based estimate", "Estimation error by query type — Our method vs TPC",
                   "robust", f"sweep_identifiability_{name}_error.png")

        # Nombre de tests CI, une figure par méthode
        for m, col, cfg in (("ours", "tests_sid", ours), ("tpc", "tests_tpc", tpc)):
            plot_sweep(df, name, [dict(value_col=col, **cfg)],
                       f"Number of CI tests ({cfg['method_label']})",
                       f"Number of CI tests by query type — {cfg['method_label']}",
                       "robust", f"sweep_identifiability_{name}_nb_tests_{m}.png")


EXPERIMENT = "sweep"  # "sweep" | "balanced" | "global"

if __name__ == "__main__":
    {"sweep": main_sweep, "balanced": main_balanced, "global": main_global}[EXPERIMENT]()

# %%