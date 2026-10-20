# %%
import numpy as np
from pyciphod.utils.scms.dynamic_scm import create_random_linear_dt_dynamic_scm
from pyciphod.utils.time_series.data_format import DTimeVar
from pyciphod.causal_discovery.local.s_identifiability import sIdentifiability, ConstructAdjSet

ZERO = ConstructAdjSet.ZERO
NOT_S_ORIENTABLE = ConstructAdjSet.NOT_S_ORIENTABLE

# --- Génération SCM, FT-DAG, SCG et données
seed, p, p_edge, gamma_max, alpha = 12, 5, 0.3, 1, 0.05
w = 1 
ORACLE = False

scm = create_random_linear_dt_dynamic_scm(
    num_ts=p, p_edge=p_edge, causally_stationary=True,
    max_delay=gamma_max, allow_instantaneous=True,
    allow_unmeasured_confounding=False,
    weight_low=-w, weight_high=w, intercept_low=0, intercept_high=0,
    u_dist=lambda: np.random.normal(0, 1), seed=seed,
)
ft_dag = scm[0].induced_ft_dag()
scg = ft_dag.get_summary_causal_graph()
lag_max = scg.get_lag_max()
t_max = ft_dag.t_max
assert t_max == lag_max

data_full = scm[0].generate_causally_stationary_time_series_from_latest_mechanisms(
    n_timepoints=5000, burn_in=200, include_latent=False,
    time_series_format="ts_lagged", seed=seed,
)
data = data_full[[c for c in data_full.columns if 0 <= -c.time <= lag_max]].copy()

sid = sIdentifiability(scg)

# --- Vérité terrain (parents / descendants dans le FT-DAG), temps data 0 <-> FT-DAG t_max
pa_true, ch_true = {}, {}
for u, v in ft_dag.get_directed_edges():
    pa_true.setdefault(v, set()).add(u)
    ch_true.setdefault(u, set()).add(v)


def to_ft(v: DTimeVar) -> DTimeVar:
    return DTimeVar(v.get_name(), v.get_time() + t_max)


def de_true(v: DTimeVar) -> set:
    """Descendants de v dans le FT-DAG (v inclus)."""
    seen, stack = {v}, [v]
    while stack:
        for c in ch_true.get(stack.pop(), ()):
            if c not in seen:
                seen.add(c)
                stack.append(c)
    return seen


def check(query: str, X: DTimeVar, Y: DTimeVar, res) -> tuple:
    """
    Vérifie la sortie de ConstructAdjSet contre le vrai FT-DAG.
    TE  : A contient Pa(X) et aucun descendant de X.
    CDE : A contient Pa(Y) \\ {X} et aucun descendant de Y.
    Retourne (ok, raison).
    """
    Xf, Yf = to_ft(X), to_ft(Y)
    if res == NOT_S_ORIENTABLE:
        return False, "non s-orientable"
    if res == ZERO:
        ok = Yf not in de_true(Xf) if query == "TE" else Xf not in pa_true.get(Yf, set())
        return ok, "" if ok else "ZERO à tort"

    A = {to_ft(a) for a in res}
    target = Xf if query == "TE" else Yf
    required = pa_true.get(target, set()) - {Xf} if query == "CDE" else pa_true.get(target, set())
    missing = required - A
    desc = A & de_true(target)
    if missing or desc:
        return False, f"parents manquants : {sorted(map(str, missing))} ; descendants inclus : {sorted(map(str, desc))}"
    return True, ""


# --- Requêtes : TE (lag 0 seulement) et CDE (lags 0..gamma_max)
names = sorted(scg.get_vertices())
queries = [("TE", 0)] + [("CDE", l) for l in range(0, gamma_max + 1)]

stats = {}  # (query, label) -> [valides, total]
total_tests = 0

for query, lag in queries:
    for nX in names:
        for nY in names:
            if lag == 0 and nX == nY:
                continue
            X, Y = DTimeVar(nX, -lag), DTimeVar(nY, 0)

            s_id = sid.is_s_identifiable(query, X, Y)

            cas = ConstructAdjSet(data, scg, alpha=alpha, oracle=ORACLE, true_ftdag=ft_dag)
            res = cas.run(query, X, Y)
            n_tests = cas.nb_ci_tests
            total_tests += n_tests

            ok, reason = check(query, X, Y, res)
            label = "s-id" if s_id else "non s-id"
            # non s-identifiable : l'aveu "non s-orientable" est aussi acceptable
            good = ok or (not s_id and res == NOT_S_ORIENTABLE)

            s = stats.setdefault((query, label), [0, 0])
            s[0] += good
            s[1] += 1

            shown = res if isinstance(res, str) else "{" + ", ".join(map(str, sorted(res))) + "}"
            print(f"{query:3} | {str(X):8} -> {str(Y):6} | {label:8} | CI : {n_tests:3} | "
                  f"{'OK' if good else 'ERREUR':6} | {shown}" + (f" | {reason}" if not good else ""))

print(f"\nTests CI effectués : {total_tests}")
print("\nPar requête et verdict a priori (corrects / total) :")
for (q, lab), (good, tot) in sorted(stats.items()):
    print(f"  {q:3} {lab:8} : {good}/{tot}")

# %%