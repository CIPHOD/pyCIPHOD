# %% Sweeps: our method (OrientInstantaneousEdge) vs TPC

import itertools
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

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
SAVE_OUTPUTS = True      # write CSVs / PNGs / TEX to OUTPUT_DIR
SHOW_PLOTS = True
OUTPUT_DIR = "outputs"
BURN_IN = 200
ORACLE = False
N_JOBS = -1              # >1: evaluate edges in parallel (needs joblib); -1: all cores

# seed=None -> not reproducible; seed=<int> -> fully reproducible
DEFAULT_PARAMS = dict(p=10, p_edge=0.4, gamma_max=2, alpha=0.05, n_timepoints=10000, seed=None)

CASE_LABELS = {
    0: "0: not adjacent",
    1: "1: already oriented in the SCG (single direction)",
    2: "2: no self-loop -> orientation via lag test",
    3: "3: different parents -> orientation via Z",
}
CASE_LABELS_TEX = {
    0: "0: not adjacent",
    1: "1: already oriented in the SCG",
    2: "2: no self-loop (lag test)",
    3: "3: different parents (via $Z$)",
}

METHOD_MARKERS = {"ours": "o", "tpc": "s"}                       # matplotlib
METHOD_MARKERS_TEX = {"ours": ("*", "o"), "tpc": ("square*", "square")}  # (filled, hollow)


# ----------------------------------------------------------------------------
# Small helpers (saving, parallelism)
# ----------------------------------------------------------------------------

def _out(name):
    """Path inside OUTPUT_DIR (directory created on demand)."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return os.path.join(OUTPUT_DIR, name)


def maybe_to_csv(df, name):
    if SAVE_OUTPUTS:
        path = _out(name)
        df.to_csv(path, index=False, na_rep="nan")
        print(f"  -> saved to {path}")


def maybe_savefig(fig, name):
    if SAVE_OUTPUTS and name is not None:
        path = _out(name)
        fig.savefig(path, dpi=150, bbox_inches="tight")
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
    ft_dag = scm.induced_ft_dag()
    return scm, ft_dag, ft_dag.get_summary_causal_graph()


def _generate_data(scm, scg, params, seed):
    np.random.seed(seed)
    lag_max = scg.get_lag_max()
    data = scm.generate_causally_stationary_time_series_from_latest_mechanisms(
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


def run_sweep(param_name, values, base_params, n_rep_per_case=100, use_tpc=False):
    """Balanced experiment for each value of `param_name` -> long DataFrame (1 row per edge)."""
    dfs = []
    for val in values:
        print(f"[sweep {param_name}={val}]")
        df = run_balanced_experiment(n_rep_per_case, {**base_params, param_name: val}, use_tpc=use_tpc)
        df["param_name"], df["param_value"] = param_name, val
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


# ----------------------------------------------------------------------------
# Figure data (shared by matplotlib and LaTeX exports)
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


def _median_iqr(data):
    """(median, Q25, Q75): robust to extreme values."""
    return (np.median(data), *np.percentile(data, [25, 75]))


def _figure_points(df, series, mode):
    """Everything needed to draw one figure.

    Returns (values, cases, points) where points[(case, series_index)] is a list of dicts
    {i, x, y, err_minus, err_plus, param_value, n}; x is the dodged horizontal position
    (category index i + offsets by case and by method)."""
    summary = _wilson_ci if mode == "binomial" else _median_iqr
    cases = [int(c) for c in sorted(df["target_case"].dropna().unique())]
    values = sorted(df["param_value"].unique())
    case_w = 0.8 / len(cases)
    method_w = case_w / len(series)

    points = {}
    for ci, case in enumerate(cases):
        for mi, s in enumerate(series):
            rows = []
            for i, val in enumerate(values):
                data = df.loc[(df["param_value"] == val) & (df["target_case"] == case),
                              s["value_col"]].dropna().values
                if len(data) == 0:
                    continue
                c, lo, hi = summary(data.astype(float))
                rows.append(dict(
                    i=i,
                    x=i + (ci - (len(cases) - 1) / 2) * case_w + (mi - (len(series) - 1) / 2) * method_w,
                    y=c, err_minus=max(0.0, c - lo), err_plus=max(0.0, hi - c),
                    param_value=val, n=len(data),
                ))
            points[(case, mi)] = rows
    return values, cases, points


def _plot_matplotlib(values, cases, series, points, param_name, ylabel, title, mode, out_name):
    colors = plt.cm.tab10.colors
    fig, ax = plt.subplots(figsize=(10, 6))
    for ci, case in enumerate(cases):
        for mi, s in enumerate(series):
            rows = points[(case, mi)]
            if not rows:
                continue
            marker = METHOD_MARKERS.get(s.get("method_key"), "^")
            color = colors[ci % len(colors)]
            fill = "full" if mi == 0 else "none"
            ax.errorbar([r["x"] for r in rows], [r["y"] for r in rows],
                        yerr=[[r["err_minus"] for r in rows], [r["err_plus"] for r in rows]],
                        fmt=marker, color=color, ecolor=color, elinewidth=1.3, capsize=3,
                        markersize=6, alpha=0.9, fillstyle=fill, markeredgewidth=1.3)
            ax.plot([], [], linestyle="", marker=marker, color=color, fillstyle=fill, markeredgewidth=1.3,
                    label=f"{CASE_LABELS.get(case, case)} — {s['method_label']}")

    ax.set_xticks(range(len(values)))
    ax.set_xticklabels([str(v) for v in values])
    ax.set_xlabel(param_name)
    ax.set_ylabel(ylabel)
    if mode == "binomial":
        ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"{title} vs {param_name}")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, borderaxespad=0.)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    maybe_savefig(fig, out_name)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


# ----------------------------------------------------------------------------
# LaTeX (pgfplots) export
# ----------------------------------------------------------------------------

def _tex_escape(s):
    return str(s).replace("_", r"\_").replace("—", "--")


def _series_key(case, series_entry):
    """Column-name suffix, letters/digits only (safe for pgfplots), e.g. 'c2ours'."""
    return f"c{case}{series_entry['method_key']}"


def _export_figure_data(values, cases, series, points, stem):
    """Wide CSV: one row per category index `i`, and for each (case, method) the columns
    x<key>, y<key>, em<key> (error down), ep<key> (error up), n<key>. NaN where no data."""
    table = pd.DataFrame({"i": range(len(values)), "param_value": values})
    for case in cases:
        for mi, s in enumerate(series):
            key = _series_key(case, s)
            for col in ("x", "y", "em", "ep", "n"):
                table[f"{col}{key}"] = np.nan
            for r in points[(case, mi)]:
                table.loc[r["i"], [f"x{key}", f"y{key}", f"em{key}", f"ep{key}", f"n{key}"]] = \
                    [r["x"], r["y"], r["err_minus"], r["err_plus"], r["n"]]
    maybe_to_csv(table, f"{stem}_data.csv")


def _export_figure_tex(values, cases, series, param_name, ylabel, title, mode, stem):
    """Standalone pgfplots document reading `<stem>_data.csv` (same folder).
    Compile with `pdflatex <stem>.tex` from OUTPUT_DIR, or copy the tikzpicture
    into your paper (needs \\usepackage{pgfplots} and \\pgfplotsset{compat=1.17})."""
    if not SAVE_OUTPUTS:
        return
    tab10 = [to_hex(c)[1:].upper() for c in plt.cm.tab10.colors]
    L = [r"\documentclass[tikz,border=5pt]{standalone}",
         r"\usepackage{pgfplots}", r"\pgfplotsset{compat=1.17}"]
    for ci, _ in enumerate(cases):
        L.append(rf"\definecolor{{case{ci}}}{{HTML}}{{{tab10[ci % len(tab10)]}}}")
    L += [r"\begin{document}", r"\begin{tikzpicture}", r"\begin{axis}["]
    opts = [
        "width=14cm, height=8cm",
        f"xlabel={{{_tex_escape(param_name)}}}",
        f"ylabel={{{_tex_escape(ylabel)}}}",
        f"title={{{_tex_escape(title)} vs {_tex_escape(param_name)}}}",
        f"xtick={{{','.join(str(i) for i in range(len(values)))}}}",
        f"xticklabels={{{','.join(str(v) for v in values)}}}",
        f"xmin=-0.6, xmax={len(values) - 0.4}",
        "ymajorgrids, grid style={gray!30}",
        "unbounded coords=discard",
        r"legend style={at={(1.02,0.5)}, anchor=west, font=\scriptsize}",
        "legend cell align=left",
    ]
    if mode == "binomial":
        opts.append("ymin=-0.05, ymax=1.05")
    L.append("  " + ",\n  ".join(opts))
    L.append("]")

    for ci, case in enumerate(cases):
        for mi, s in enumerate(series):
            key = _series_key(case, s)
            filled, hollow = METHOD_MARKERS_TEX.get(s["method_key"], ("triangle*", "triangle"))
            mark = filled if mi == 0 else hollow
            L.append(
                rf"\addplot[only marks, color=case{ci}, mark={mark}, mark options={{solid}}, "
                rf"error bars/.cd, y dir=both, y explicit] "
                rf"table[col sep=comma, x=x{key}, y=y{key}, y error plus=ep{key}, y error minus=em{key}] "
                rf"{{{os.path.basename(stem)}_data.csv}};"
            )
            L.append(rf"\addlegendentry{{{CASE_LABELS_TEX.get(case, case)} -- {_tex_escape(s['method_label'])}}}")
    L += [r"\end{axis}", r"\end{tikzpicture}", r"\end{document}"]
    path = _out(f"{stem}.tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"  -> saved to {path}")


# ----------------------------------------------------------------------------
# One figure = PNG + data CSV + pgfplots TEX
# ----------------------------------------------------------------------------

def make_figure(df, param_name, series, ylabel, title, mode, stem):
    series = [s for s in series if s["value_col"] in df.columns and df[s["value_col"]].notna().any()]
    if not series:
        print("  [warning] no valid series to plot, skipping figure")
        return
    values, cases, points = _figure_points(df, series, mode)
    _plot_matplotlib(values, cases, series, points, param_name, ylabel, title, mode, f"{stem}.png")
    if SAVE_OUTPUTS:
        _export_figure_data(values, cases, series, points, stem)
        _export_figure_tex(values, cases, series, param_name, ylabel, title, mode, stem)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main_sweep(n_rep_per_case=100, use_tpc=True):
    base_params = dict(p=5, p_edge=0.4, gamma_max=1, alpha=0.05, n_timepoints=5000, seed=123)
    sweeps = {
        "p_edge": [0.2, 0.4, 0.6, 0.8],
        "p": [5, 10, 15, 20],
        "n_timepoints": [500, 1000, 5000, 10000, 20000],
        "gamma_max": [1, 2, 3, 4],
    }
    ours = dict(method_label="Our method", method_key="ours")
    tpc = dict(method_label="TPC", method_key="tpc")

    for name, values in sweeps.items():
        df = run_sweep(name, values, base_params, n_rep_per_case=n_rep_per_case, use_tpc=use_tpc)
        maybe_to_csv(df, f"sweep_{name}.csv")  # raw per-edge results

        series = [dict(value_col="our_correct", **ours)] + \
                 ([dict(value_col="tpc_correct", **tpc)] if use_tpc else [])
        make_figure(df, name, series,
                    ylabel="Correct proportion (per edge)",
                    title="Correct proportion by case" + (" -- Our method vs TPC" if use_tpc else ""),
                    mode="binomial", stem=f"sweep_{name}_correctness")

        for m, col, cfg in (("ours", "our_nb_tests", ours), ("tpc", "tpc_nb_tests", tpc)):
            if m == "tpc" and not use_tpc:
                continue
            make_figure(df, name, [dict(value_col=col, **cfg)],
                        ylabel=f"Number of CI tests ({cfg['method_label']})",
                        title=f"Number of CI tests by case -- {cfg['method_label']}",
                        mode="robust", stem=f"sweep_{name}_nb_tests_{m}")


if __name__ == "__main__":
    main_sweep()

# %%