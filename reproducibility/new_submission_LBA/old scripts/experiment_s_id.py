# %% Effet causal estimé : ensemble d'ajustement du DAG vs s-identifiabilité vs TPC
import itertools
import numpy as np
import pandas as pd

from pyciphod.utils.scms.dynamic_scm import create_random_linear_dt_dynamic_scm
from pyciphod.utils.time_series.data_format import DTimeVar
from pyciphod.causal_discovery.local.orientation_using_scg import sIdentifiability
from pyciphod.causal_discovery.basic.ts_constraint_based import TsPC
from pyciphod.utils.graphs.background_knowledge import BackgroundKnowledge
from pyciphod.utils.stat_tests.independence_tests import FisherZTest

# %% Paramètres
N_SIM = 100                       # une simulation = un DAG différent
P, P_EDGE, GMAX_GEN = 10, 0.3, 1
ALPHA, N_TIMEPOINTS, BURN_IN = 0.05, 10000, 200
ORACLE = False                    # True : d-séparation exacte ; False : tests CI sur les données
CI = FisherZTest


# %% Simulation
def simulate(seed):
    np.random.seed(seed)
    w = 1 / (P * P_EDGE * GMAX_GEN)
    scm = create_random_linear_dt_dynamic_scm(
        num_ts=P, p_edge=P_EDGE, causally_stationary=True, max_delay=GMAX_GEN,
        allow_instantaneous=True, allow_unmeasured_confounding=False,
        weight_low=-w, weight_high=w, intercept_low=0.0, intercept_high=0.0,
        u_dist=lambda: np.random.normal(0, 1), seed=seed,
    )
    ft_dag = scm.induced_ft_dag()
    scg = ft_dag.get_summary_causal_graph()
    gmax = scg.get_lag_max()
    full = scm.generate_causally_stationary_time_series_from_latest_mechanisms(
        n_timepoints=N_TIMEPOINTS, burn_in=BURN_IN, include_latent=False,
        time_series_format="ts_lagged", seed=seed,
    )
    # 2*gmax lags : l'ensemble d'ajustement de l'ATE remonte jusqu'à t-γ-gmax
    data = full[[c for c in full.columns if -c.time <= 2 * gmax]].copy()
    return ft_dag, scg, gmax, data


# %% Utilitaires
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
    X = np.column_stack([data[c].to_numpy() for c in cols] + [np.ones(len(data))])
    return np.linalg.lstsq(X, data[outcome].to_numpy(), rcond=None)[0][0]


def bk_from_scg(scg, gmax):
    """Connaissance a priori pour TPC : absences d'arêtes et orientations du SCG."""
    bk, nodes = BackgroundKnowledge(), list(scg.get_vertices())
    lags = list(itertools.product(range(gmax + 1), repeat=2))
    for x, y in itertools.combinations(nodes, 2):
        if y not in scg.get_adjacencies(x):
            for k, l in lags:
                bk.add_forbidden_edge(DTimeVar(x, -k), DTimeVar(y, -l))
                bk.add_forbidden_edge(DTimeVar(y, -k), DTimeVar(x, -l))
        else:
            for a, b in ((x, y), (y, x)):
                if a in scg.get_parents(b) and a not in scg.get_children(b):
                    bk.add_mandatory_orientation(DTimeVar(a, 0), DTimeVar(b, 0))
    for x in nodes:
        if x not in scg.get_adjacencies(x):     # pas d'auto-boucle
            for k, l in lags:
                if k != l:
                    bk.add_forbidden_edge(DTimeVar(x, -k), DTimeVar(x, -l))
    return bk


def draw_queries(rng, ft_dag, scg, gmax):
    """Une requête ATE (Y descendant de X) et une requête CDE (X_{t-γ} parent de Y_t)."""
    nodes = list(scg.get_vertices())
    queries = []

    ate = [(x, y) for x, y in itertools.permutations(nodes, 2)
           if y in scg.get_descendants(x)]
    if ate:
        x, y = ate[rng.integers(len(ate))]
        queries.append(("ATE", DTimeVar(x, -int(rng.integers(gmax + 1))), y))

    # CDE : on tire une arête X_{t-γ} -> Y_t du FT-DAG (avec X ≠ Y), donc Y est bien enfant de X
    cde = [(T, y) for y in nodes for T in parents_dag(ft_dag, gmax, y) if T.get_name() != y]
    if cde:
        T, y = cde[rng.integers(len(cde))]
        queries.append(("CDE", T, y))
    return queries


# %% Une simulation : un DAG, un run TPC, une requête ATE et une requête CDE
def run_seed(seed):
    rng = np.random.default_rng(seed)
    ft_dag, scg, gmax, data = simulate(seed)

    # TPC : une seule découverte par DAG, sur la fenêtre [-gmax, 0]
    tpc = TsPC(ALPHA, ci_test=CI, oracle=ORACLE, true_ftdag=ft_dag,
               background_knowledge=bk_from_scg(scg, gmax))
    tpc.run(data[[c for c in data.columns if -c.time <= gmax]])

    rows = []
    for query, T, y in draw_queries(rng, ft_dag, scg, gmax):
        Y = DTimeVar(y, 0)
        # ATE : Pa(X_{t-γ}) ; CDE : Pa(Y_t)
        name, dt = (T.get_name(), T.get_time()) if query == "ATE" else (y, 0)

        adj_dag = shift(parents_dag(ft_dag, gmax, name), dt)      # (i) DAG
        adj_tpc = shift(parents_tpc(tpc.g_hat, name), dt)         # (iii) TPC
        sid = sIdentifiability(data, T, Y, query, scg, ALPHA, CI, ORACLE, ft_dag)
        adj_sid = sid.get_adj_set()                               # (ii) s-identifiabilité

        row = dict(seed=seed, query=query, T=T, Y=Y, identified=adj_sid is not None,
                   tests_sid=sid.nb_ci_tests, tests_tpc=tpc.nb_ci_tests)
        sets = {"dag": adj_dag, "sid": adj_sid, "tpc": adj_tpc}
        if all(s is not None for s in sets.values()):
            sets = {k: s - {T, Y} for k, s in sets.items()}
            if all(c in data.columns for s in sets.values() for c in s):
                for k, s in sets.items():
                    row[f"est_{k}"] = ols_effect(data, Y, T, s)
        rows.append(row)
    return rows


rows = []
for seed in range(2020, 2020 + N_SIM):
    rows += run_seed(seed)
    print(f"sim {seed} done")
df = pd.DataFrame(rows)

# %% Résultats : erreur quadratique par rapport à l'estimation basée sur le DAG
ok = df.dropna(subset=["est_dag", "est_sid", "est_tpc"]).copy()
ok["se_sid"] = (ok["est_sid"] - ok["est_dag"]) ** 2
ok["se_tpc"] = (ok["est_tpc"] - ok["est_dag"]) ** 2

print(f"\n{len(ok)}/{len(df)} cas comparables ({df['identified'].mean():.0%} s-identifiés)\n")
print(ok.groupby("query").agg(n=("se_sid", "size"),
                              MSE_sid=("se_sid", "mean"), MSE_tpc=("se_tpc", "mean")))

# s-id : moyenne par requête ; TPC : une découverte par DAG
print(f"\nTests CI moyens par requête (s-id) : {df['tests_sid'].mean():.1f}")
print(f"Tests CI moyens par DAG (TPC)      : {df.groupby('seed')['tests_tpc'].first().mean():.1f}")


###############################################################
###############################################################
###############################################################
###############################################################
###############################################################


# %% Debug d'UNE itération, à exécuter bloc par bloc (# %%) ou ligne par ligne
# Prérequis : avoir exécuté les cellules "imports", "paramètres", "simulation" et "utilitaires"
# de compare_adjustment_v2.py (mais PAS la boucle finale ni run_seed).

def fmt(v):
    """Affiche un DTimeVar de façon lisible : X0(t-1), X3(t)."""
    t = v.get_time()
    return f"{v.get_name()}(t{t:+d})" if t else f"{v.get_name()}(t)"

def fmts(nodes):
    return sorted(fmt(v) for v in nodes)


# %% 1) Simulation : données + graphes
seed = 20200
rng = np.random.default_rng(seed)
ft_dag, scg, gmax, data = simulate(seed)

print("gmax (retard max du SCG) :", gmax)
print("Forme des données        :", data.shape,
      f"(attendu : {N_TIMEPOINTS} lignes environ, {P * (2 * gmax + 1)} colonnes)")
print("Colonnes                 :", fmts(data.columns))
data.head()


# %% 2) Vérifications sur les données
# (a) Les colonnes retardées sont-elles bien des décalages des colonnes courantes ?
x0 = list(scg.get_vertices())[0]
cur, lag1 = data[DTimeVar(x0, 0)].to_numpy(), data[DTimeVar(x0, -1)].to_numpy()
print(f"{x0}(t-1)[t] == {x0}(t)[t-1] ?", np.allclose(lag1[1:], cur[:-1]))

# (b) Ordre de grandeur / NaN
print("NaN :", int(data.isna().sum().sum()))
print("Écart-type par colonne (min, max) :", data.std().min().round(2), data.std().max().round(2))


# %% 3) Graphes vrais : SCG et FT-DAG
print("SCG : enfants de chaque série")
for x in scg.get_vertices():
    print(f"  {x} -> {sorted(scg.get_children(x))}")

true_edges = [(DTimeVar(u.get_name(), u.get_time() - gmax),
               DTimeVar(w.get_name(), w.get_time() - gmax))
              for u, w in ft_dag.get_directed_edges()]
print(f"\nFT-DAG : {len(true_edges)} arêtes (temps recentré : t courant = 0). Exemples :")
for u, w in true_edges[:8]:
    print(f"  {fmt(u)} -> {fmt(w)}")


# %% 4) TPC : une découverte pour ce DAG
bk = bk_from_scg(scg, gmax)
tpc = TsPC(ALPHA, ci_test=CI, oracle=ORACLE, true_ftdag=ft_dag, background_knowledge=bk)
tpc.run(data[[c for c in data.columns if -c.time <= gmax]])

print("Tests CI TPC :", tpc.nb_ci_tests)
est_edges = set(tpc.g_hat.get_directed_edges())
print("Arêtes orientées estimées :", len(est_edges))

# Comparaison aux vraies arêtes qui arrivent en t (seules lues pour l'ajustement)
true_in = {(u, w) for u, w in true_edges if w.get_time() == 0}
est_in = {(u, w) for u, w in est_edges if w.get_time() == 0}
print(f"Arêtes vers t : vraies={len(true_in)}, estimées={len(est_in)}, "
      f"correctes={len(true_in & est_in)}, en trop={len(est_in - true_in)}, "
      f"manquantes={len(true_in - est_in)}")


# %% 5) Tirage des requêtes
queries = draw_queries(rng, ft_dag, scg, gmax)
for q, T, y in queries:
    print(f"{q} : traitement {fmt(T)}  ->  outcome {y}(t)")


# %% 6) Une requête en détail (change l'indice 0/1 pour passer d'ATE à CDE)
query, T, y = queries[0]
Y = DTimeVar(y, 0)
print(f"Requête {query} : {fmt(T)} -> {fmt(Y)}")

# Variable dont on prend les parents : X_{t-γ} pour l'ATE, Y_t pour la CDE
name, dt = (T.get_name(), T.get_time()) if query == "ATE" else (y, 0)
print(f"On ajuste sur les parents de {name} décalés de {dt}")

adj_dag = shift(parents_dag(ft_dag, gmax, name), dt)
adj_tpc = shift(parents_tpc(tpc.g_hat, name), dt)
print("\n(i)   Ajustement DAG :", fmts(adj_dag))
print("(iii) Ajustement TPC :", fmts(adj_tpc))


# %% 7) s-identifiabilité
sid = sIdentifiability(data, T, Y, query, scg, ALPHA, CI, ORACLE, ft_dag)
adj_sid = sid.get_adj_set()
print("Identifiable :", adj_sid is not None)
print("(ii)  Ajustement s-id :", None if adj_sid is None else fmts(adj_sid))
print("Tests CI s-id :", sid.nb_ci_tests)


# %% 8) Estimation OLS avec chaque ensemble
sets = {"dag": adj_dag, "sid": adj_sid, "tpc": adj_tpc}
if all(s is not None for s in sets.values()):
    sets = {k: s - {T, Y} for k, s in sets.items()}       # on retire T et Y des contrôles
    missing = {c for s in sets.values() for c in s if c not in data.columns}
    print("Contrôles absents des données :", fmts(missing) or "aucun")
    est = {k: ols_effect(data, Y, T, s) for k, s in sets.items()}
    est["naïf (sans ajustement)"] = ols_effect(data, Y, T, set())
    for k, v in est.items():
        print(f"  {k:24s} {v: .5f}")
    print("\nÉcart quadratique vs DAG :  sid =", (est["sid"] - est["dag"]) ** 2,
          "| tpc =", (est["tpc"] - est["dag"]) ** 2)
else:
    print("Cas non comparable : au moins un ensemble est None (requête non s-identifiée).")
# %%
