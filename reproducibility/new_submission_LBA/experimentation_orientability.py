# %%
from collections import Counter

import numpy as np
from pyciphod.utils.scms.dynamic_scm import create_random_linear_dt_dynamic_scm
from pyciphod.utils.time_series.data_format import DTimeVar
from pyciphod.causal_discovery.local.s_orientability import OrientEdge

NON_ADJACENT = OrientEdge.NON_ADJACENT
NOT_S_ORIENTABLE = OrientEdge.NOT_S_ORIENTABLE

# --- Génération SCM, FT-DAG, SCG et données
seed, p, p_edge, gamma_max, alpha = 1030, 7, 0.4, 1, 0.05
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

data_full = scm[0].generate_causally_stationary_time_series_from_latest_mechanisms(
    n_timepoints=5000, burn_in=200, include_latent=False,
    time_series_format="ts_lagged", seed=seed,
)
data = data_full[
    [c for c in data_full.columns if 0 <= -c.time <= lag_max]
].copy()

assert t_max == lag_max

oe = OrientEdge(data, scg, alpha=alpha, oracle=ORACLE, true_ftdag=ft_dag)

# --- Test des arêtes ayant au moins une extrémité au présent
edges = [(u, v) for u, v in ft_dag.get_directed_edges()
         if u.get_time() == t_max or v.get_time() == t_max]

ok = errors = total_tests = 0
stats = Counter()  # cas de s-orientabilité -> [correctes, total]
by_case = {}

for u, v in sorted(edges):
    A = DTimeVar(u.get_name(), u.get_time() - t_max)
    B = DTimeVar(v.get_name(), v.get_time() - t_max)

    # Test a priori (SCG seul) : (True, cas) si s-orientable, (False, None) sinon
    s_orientable, case = oe.is_s_orientable(A, B)
    case_label = str(case) if s_orientable else "non s-or."

    before = oe.nb_ci_tests
    pred = oe._run(A, B)
    n_tests = oe.nb_ci_tests - before
    total_tests += n_tests

    # Attendu : l'orientation vraie si s-orientable, sinon NOT_S_ORIENTABLE
    true = f"{A} -> {B}"
    expected = true if s_orientable else NOT_S_ORIENTABLE
    correct = pred == expected

    ok += correct
    errors += not correct
    by_case.setdefault(case_label, [0, 0])
    by_case[case_label][0] += correct
    by_case[case_label][1] += 1

    print(f"Réel : {true:25} | Prédit : {pred:25} | cas : {case_label:9} | "
          f"CI : {n_tests:3} | {'OK' if correct else 'ERREUR'}")

print(f"\nArêtes : {len(edges)} | Correctes : {ok} | Erreurs : {errors}")
print(f"Tests CI effectués : {total_tests} | Total cumulatif : {oe.nb_ci_tests}")
print("\nPar cas de s-orientabilité (correctes / total) :")
for c, (good, tot) in sorted(by_case.items()):
    print(f"  {c:9} : {good}/{tot}")

# %%