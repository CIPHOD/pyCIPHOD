# Debug pas à pas : UN seul SCM, UN seul effet. Exécuter cellule par cellule (# %%).
# Script AUTONOME : n'importe rien depuis un autre script d'expérience.
# %% 0. Imports (mêmes que le script d'expérience)
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import itertools
import time
from functools import lru_cache

import numpy as np
import pandas as pd

from pyciphod.utils.scms.dynamic_scm import create_random_linear_dt_dynamic_scm
from pyciphod.utils.time_series.data_format import DTimeVar, wide_timevar_to_lagged_df
from pyciphod.causal_discovery.local.s_orientability import OrientEdge
# Si sIdentifiability / ConstructAdjSet sont ailleurs chez toi, adapte ici :
try:
    from pyciphod.causal_discovery.local.s_identifiability import sIdentifiability, ConstructAdjSet
except ImportError:
    from pyciphod.causal_discovery.local.orientation_using_scg import sIdentifiability, ConstructAdjSet
from pyciphod.causal_discovery.basic.ts_constraint_based import TsPC
from pyciphod.utils.graphs.background_knowledge import BackgroundKnowledge
from pyciphod.utils.stat_tests.independence_tests import FisherZTest

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 200)

# %% 1. Réglages
QUERY = "ATE"            # "ATE" ou "CDE"
SEED = None              # None -> premier seed donnant un effet candidat
FORCE = None             # ex: ("X3", "X5", 1) pour forcer (nX, nY, lag)
BURN_IN = 200
MAX_EFFECT_LAG = None
MIN_ABS_EFFECT = 0.1
MAX_SPECTRAL_RADIUS = 0.95
PARAMS = dict(p=10, p_edge=0.2, gamma_max=2, alpha=0.05, n_timepoints=5000, seed=17102026)
params = dict(PARAMS)    # modifiable : params["p"] = 5, params["n_timepoints"] = 2000, ...


# %% 2. Fonctions (copie de l'expérience)
def simulate(params, seed):
    np.random.seed(seed)
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
    n = len(names)
    B, idx, p = _coef_matrices(template, names)
    inv = np.linalg.inv(np.eye(n) - B[0])
    psi = [inv]
    for h in range(1, h_max + 1):
        acc = sum((B[k] @ psi[h - k] for k in range(1, min(h, p) + 1)), np.zeros((n, n)))
        psi.append(inv @ acc)
    return psi, idx


def spectral_radius(template, names):
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
    gamma = scg.get_lag_max()
    cap = gamma if MAX_EFFECT_LAG is None else min(MAX_EFFECT_LAG, gamma)
    sid = sIdentifiability(scg)
    ok = {n: not sid._non_s_orientable_with(n) for n in scg.get_vertices()}

    template = coefficient_template(scm)
    names = sorted({n.name for n in scm[0].v})
    out = {"ATE": [], "CDE": []}
    if spectral_radius(template, names) >= MAX_SPECTRAL_RADIUS:
        return out
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


def estimate_effect(data, X, Y, adj):
    cols = [X] + sorted(set(adj) - {X, Y})
    missing = [c for c in cols + [Y] if c not in data.columns]
    if missing:
        raise KeyError(f"columns missing from data (window too small?): {missing}")
    sub = data[cols + [Y]].dropna()
    A = np.column_stack([np.ones(len(sub)), sub[cols].to_numpy(float)])
    coef = np.linalg.lstsq(A, sub[Y].to_numpy(float), rcond=None)[0]
    return float(coef[1]), len(cols) - 1


def bk_from_scg(scg, gamma_max):
    bk = BackgroundKnowledge()
    nodes = list(scg.get_vertices())
    all_kl = list(itertools.product(range(gamma_max + 1), repeat=2))
    for x, y in itertools.combinations(nodes, 2):
        if y not in set(scg.get_adjacencies(x)):
            for k, l in all_kl:
                bk.add_forbidden_edge(DTimeVar(x, -k), DTimeVar(y, -l))
                bk.add_forbidden_edge(DTimeVar(y, -k), DTimeVar(x, -l))
        else:
            for a, b in ((x, y), (y, x)):
                if a in set(scg.get_parents(b)) and a not in set(scg.get_children(b)):
                    bk.add_mandatory_orientation(DTimeVar(a, 0), DTimeVar(b, 0))
    for x in nodes:
        if x not in set(scg.get_adjacencies(x)):
            for k, l in all_kl:
                if k != l:
                    bk.add_forbidden_edge(DTimeVar(x, -k), DTimeVar(x, -l))
    return bk


# %% 3. Choix du SCM et de l'effet
def find_seed(start=0):
    for s in range(start, start + 5000):
        scm_, ft_, scg_ = simulate(params, s)
        c = candidate_effects(scm_, scg_)[QUERY]
        if c:
            return s, c
    raise RuntimeError("aucun seed trouvé")


if SEED is None:
    SEED, cands = find_seed()
scm, ft_dag, scg = simulate(params, SEED)
cands = candidate_effects(scm, scg)[QUERY]
model, coefs, intercepts = scm
gamma = scg.get_lag_max()
print(f"seed={SEED}  gamma_max(SCG)={gamma}  #candidats {QUERY}={len(cands)}")
print(pd.DataFrame(cands, columns=["nX", "nY", "lag", "truth"]).head(30))

if FORCE:
    nX, nY, lag = FORCE
    truth = next(t for (a, b, l, t) in cands if (a, b, l) == (nX, nY, lag))
else:
    nX, nY, lag, truth = cands[0]
is_te = QUERY == "ATE"
X, Y = DTimeVar(nX, -lag), DTimeVar(nY, 0)
nT, u = (nX, -lag) if is_te else (nY, 0)     # série cible et son instant
print(f"\nEffet testé : {X} -> {Y}   (cible T = {nT} à t={u})   vérité = {truth:.4f}")


# %% 4. Graphes : SCG et FT-DAG
for g, name in ((scg, "SCG"), (ft_dag, "FT-DAG")):
    try:
        g.draw_graph()
    except Exception as e:
        print(f"draw_graph() indisponible sur {name}: {e}")


# %% 5. Coefficients
template = coefficient_template(scm)
names = sorted({n.name for n in model.v})
coef_df = (pd.DataFrame([(s, t, l, w) for (s, t, l), w in template.items()],
                        columns=["src", "tgt", "lag", "coef"])
           .sort_values(["tgt", "lag", "src"]).reset_index(drop=True))
print(coef_df)
print("\nRayon spectral (doit être < 1) :", round(spectral_radius(template, names), 3))


# %% 6. Vérité terrain : deux calculs indépendants
psi, idx = impulse_responses(template, names, max(lag, 1))
truth_matrix = psi[lag][idx[nY], idx[nX]]

parents_of = {}
for (s, t, l), w in template.items():
    parents_of.setdefault(t, []).append((s, l, w))


@lru_cache(None)
def effect(src_name, src_time, tgt_name, tgt_time):
    if (src_name, src_time) == (tgt_name, tgt_time):
        return 1.0
    if tgt_time < src_time:
        return 0.0
    return sum(w * effect(src_name, src_time, s, tgt_time - l)
               for (s, l, w) in parents_of.get(tgt_name, []))


truth_paths = effect(nX, -lag, nY, 0)
direct = template.get((nX, nY, lag), 0.0)
print(f"TE matrice={truth_matrix:.4f} | TE chemins={truth_paths:.4f} | coef direct (CDE)={direct:.4f}")
print(f"vérité utilisée pour {QUERY} = {truth:.4f}")
assert np.isclose(truth_matrix, truth_paths), "les deux calculs du TE divergent !"
assert np.isclose(truth, truth_matrix if is_te else direct), "la vérité de l'expérience est incohérente"


# %% 7. Données
data = generate_data(scm, params, SEED, window=lag + gamma + 1)
print(data.shape, "| instants présents :", sorted({c.time for c in data.columns}))
print(data.head())
print(data.describe().T[["mean", "std", "min", "max"]].round(2).head(15))


# %% 8. FT-DAG oracle : parents vrais de la cible
t_max = ft_dag.t_max
T_node = DTimeVar(nT, t_max)
true_parents = {DTimeVar(a.get_name(), a.get_time() - t_max + u)
                for a, b in ft_dag.get_directed_edges() if b == T_node}
print("Parents vrais de la cible :", sorted(map(str, true_parents)))
beta_ft, k_ft = estimate_effect(data, X, Y, true_parents)
print(f"[FT-DAG] beta={beta_ft:.4f}  vérité={truth:.4f}  err_rel={abs(beta_ft-truth)/abs(truth):.3%}")


# %% 9. Notre méthode (ConstructAdjSet)
cas = ConstructAdjSet(data, scg, alpha=params["alpha"], oracle=False)
_, L, Q, R = cas._instantiate(QUERY, X, Y)
print("L (retardés)         :", sorted(map(str, L)))
print("Q (inst., non-desc.) :", sorted(map(str, Q)))
print("R (inst., desc. SCG) :", sorted(map(str, R)))

res = cas.run(QUERY, X, Y)
print("\nRésultat :", res if isinstance(res, str) else sorted(map(str, res)))
print("Nb tests CI :", cas.nb_ci_tests)
beta_our = np.nan
if not isinstance(res, str):
    print("Dans notre ensemble mais pas parent vrai :", sorted(map(str, set(res) - true_parents)))
    print("Parent vrai absent de notre ensemble     :", sorted(map(str, true_parents - set(res) - {Y})),
          "  <- doit être vide !")
    beta_our, _ = estimate_effect(data, X, Y, res)
    print(f"[OURS]   beta={beta_our:.4f}  vérité={truth:.4f}  err_rel={abs(beta_our-truth)/abs(truth):.3%}")


# %% 10. TPC avec la même connaissance du SCG
sub = data[[c for c in data.columns if -c.time <= gamma]]
tpc = TsPC(sparsity=params["alpha"], ci_test=FisherZTest,
           background_knowledge=bk_from_scg(scg, gamma), oracle=False, true_ftdag=ft_dag)
tpc.run(sub)
print("Nb tests CI TPC :", tpc.nb_ci_tests)
try:
    tpc.g_hat.draw_graph()
except Exception as e:
    print("draw_graph() indisponible sur g_hat:", e)

T0 = DTimeVar(nT, 0)
und = [e for e in tpc.g_hat.get_undirected_edges() if T0 in e]
tpc_par = {A for (A, B) in tpc.g_hat.get_directed_edges() if B == T0}
print("Arêtes non orientées incidentes à la cible :", und)
print("Parents TPC (au temps 0) :", sorted(map(str, tpc_par)))
beta_tpc = np.nan
if not und:
    adj = {DTimeVar(A.get_name(), A.get_time() + u) for A in tpc_par}
    print("Ajustement TPC décalé    :", sorted(map(str, adj)))
    print("Faux parents TPC         :", sorted(map(str, adj - true_parents)))
    print("Parents manquants TPC    :", sorted(map(str, true_parents - adj)))
    beta_tpc, _ = estimate_effect(data, X, Y, adj)
    print(f"[TPC]    beta={beta_tpc:.4f}  vérité={truth:.4f}  err_rel={abs(beta_tpc-truth)/abs(truth):.3%}")
else:
    print("TPC : échec (voisin non orienté)")


# %% 11. Comparaison finale
print(f"vérité={truth:.4f} | FT-DAG={beta_ft:.4f} | OURS={beta_our:.4f} | TPC={beta_tpc:.4f}")
# %%
