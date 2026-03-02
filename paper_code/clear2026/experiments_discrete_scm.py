
#%%
# Required packages
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.linear_model import LinearRegression 
import matplotlib.pyplot as plt
import seaborn as sns
import time
import sys
import os
import pprint
from itertools import combinations
from scipy.special import expit  # sigmoid

import random

# Set seed for reproducibility

# General setting : 

n_exp = 100


# Imports
from locPC_CDE import runLocPC_CDE, LocPC
from dags_generator import random_DAG_nonidentifiable_CDE, random_DAG_identifiable_CDE
from baselines.Gupta_codes.ldecc import LDECCAlgorithm
from baselines.Gupta_codes.pc_alg import PCAlgorithm
from baselines.pyCausalFS.LSL.MBs.CMB.CMB import CMB
from baselines.pyCausalFS.LSL.MBs.MBbyMB import MBbyMB


# Get oracle 

from itertools import combinations

def simulate_binarySCM_logistic_from_dag(dag, nb_obs=1, coef_range=(-5, 5), bias_range=(0, 0)):
    """
    Simulates data from a binary stochastic Structural Causal Model (SCM) defined by a DAG.
    Each variable is binary and generated via logistic regression from its parents.
    
    Parameters:
    - dag: networkx.DiGraph (nodes should be ordered or sortable topologically)
    - nb_obs: number of observations to generate
    - coef_range: range for random edge weights
    - bias_range: range for node biases (intercepts)
    
    Returns:
    - coef_df: DataFrame of weights (p x p) where entry (j, i) is the effect of X_i on X_j
    - bias_df: DataFrame of biases/intercepts for each variable
    - data_df: DataFrame of generated binary data (nb_obs x p)
    """
    v = list(nx.topological_sort(dag))
    p = len(v)
    
    coef = np.zeros((p, p))  # coef[j, i] is effect of X_i on X_j
    bias = np.random.uniform(bias_range[0], bias_range[1], size=p)
    
    low, high = coef_range
    for j in range(p):
        pa_v = list(dag.predecessors(v[j]))
        for i in range(p):
            if v[i] in pa_v:
                while np.abs(coef[j, i]) < 0.2:
                    coef[j, i] = np.random.uniform(low, high)
    
    data = np.zeros((nb_obs, p), dtype=int)
    name_to_idx = {name: idx for idx, name in enumerate(v)}
    
    for obs in range(nb_obs):
        x = np.zeros(p)
        for j in range(p):
            pa_v = list(dag.predecessors(v[j]))
            pa_indices = [name_to_idx[pn] for pn in pa_v]
            linear_input = np.dot(coef[j, pa_indices], x[pa_indices]) + bias[j]
            prob = expit(linear_input)
            x[j] = np.random.binomial(1, prob)
        data[obs, :] = x
    
    coef_df = pd.DataFrame(coef, index=v, columns=v)
    bias_df = pd.Series(bias, index=v, name="bias")
    data_df = pd.DataFrame(data, columns=v)
    
    return data_df




#### BASELINES


# Algorithms
def ldecc_CDE(data, treatment, outcome):
    start_time = time.time()
    Y, X = outcome, treatment

    ldecc_alg = LDECCAlgorithm(treatment_node=Y, outcome_node=Y, use_ci_oracle=False, ci_test="gsq")
    ldecc_res = ldecc_alg.run(data)

    unoriented = ldecc_res['unoriented']
    identifiability = False
    if X in ldecc_res['tmt_children'] or X not in (ldecc_res['tmt_parents'] | ldecc_res['tmt_children']) or len(unoriented) == 0:
        identifiability = True 

    adjustment_set = list(ldecc_res['tmt_parents']) if identifiability else None
    exec_time = time.time() - start_time

    return {
        'identifiability': identifiability,
        'exec_time': exec_time,
        'adjustment_set': adjustment_set,
        'nb_CI_tests': ldecc_alg.nb_ci_tests
    }


def PC_CDE(data, treatment, outcome):
    Y, X = outcome, treatment

    pc_alg = PCAlgorithm(treatment_node=Y, outcome_node=Y, use_ci_oracle=False, ci_test="gsq")
    start_time = time.time()
    pc_res_cpdag, pc_res_nb_tests = pc_alg.run(data)

    identifiability = False 
    if X in set(pc_res_cpdag.successors(Y)) or \
       (X not in set(pc_res_cpdag.successors(Y)) | set(pc_res_cpdag.predecessors(Y))) or \
       all(p not in set(pc_res_cpdag.successors(Y)) for p in pc_res_cpdag.predecessors(Y)):
        identifiability = True

    adjustment_set = list(set(pc_res_cpdag.predecessors(Y))) if identifiability else None
    exec_time = time.time() - start_time

    return {
        'identifiability': identifiability,
        'exec_time': exec_time,
        'adjustment_set': adjustment_set,
        'nb_CI_tests': pc_res_nb_tests
    }

def replace_indices_by_names(data, cmb_res):
    # Helper function to convert indices list to column names list
    def indices_to_names(indices):
        if not indices:  # empty list
            return []
        return [data.columns[i] for i in indices]

    parents = indices_to_names(cmb_res[0])
    children = indices_to_names(cmb_res[1])
    pc = indices_to_names(cmb_res[2])
    unoriented = indices_to_names(cmb_res[3])

    return parents, children, pc, unoriented


def CMB_CDE(data, treatment, outcome):
    X = data.columns.get_loc(treatment)
    Y = data.columns.get_loc(outcome)
    start_time = time.time()

    cmb_res = CMB(data, Y, 0.05, is_discrete=True)
    parents, children, pc, unoriented = replace_indices_by_names(data, cmb_res)

    identifiability = False
    if (treatment in children) or (treatment not in pc) or (len(unoriented) == 0):
        identifiability = True

    adjustment_set = list(parents) if identifiability else None
    exec_time = time.time() - start_time

    return {
        'identifiability': identifiability,
        'exec_time': exec_time,
        'adjustment_set': adjustment_set,
        'nb_CI_tests': cmb_res[4]
    }

def MBbyMB_CDE(data, treatment, outcome):
    X = data.columns.get_loc(treatment)
    Y = data.columns.get_loc(outcome)
    start_time = time.time()

    mbbymb_res = MBbyMB(data, Y, 0.05, is_discrete=True)
    parents, children, pc, unoriented = replace_indices_by_names(data, mbbymb_res)

    identifiability = False
    if (treatment in children) or (treatment not in pc) or (len(unoriented) == 0):
        identifiability = True

    adjustment_set = list(parents) if identifiability else None
    exec_time = time.time() - start_time

    return {
        'identifiability': identifiability,
        'exec_time': exec_time,
        'adjustment_set': adjustment_set,
        'nb_CI_tests': mbbymb_res[4]
    }


def MMBbyMMB_CDE(data, treatment, outcome):
    # récupérer les indices des colonnes treatment et outcome
    X = data.columns.get_loc(treatment)
    Y = data.columns.get_loc(outcome)
    data_np = data.to_numpy()
    start_time = time.time()

    learn_graph = MMB_by_MMB(Data = data_np, 
                         target = Y, 
                         alpha = 0.05, 
                         test = "gsq", 
                         p = data.shape[1],
                         maxK = data.shape[1])
    
    parents, bidir, bicircle, ci_test = learn_graph.mmb_by_mmb()
    
    
    identifiability = not (len(bidir) != 0 or len(bicircle != 0))

    if identifiability:
        parents = data.columns[parents]
        adjustment_set = list(parents)
        if X not in adjustment_set:
            estimated_linear_DE = 0
        else:
            model = LinearRegression()
            X_adjust = data[adjustment_set]
            y_target = data[outcome]
            model.fit(X_adjust, y_target)
            estimated_linear_DE = model.coef_[adjustment_set.index(treatment)]
    else:
        estimated_linear_DE = None
        adjustment_set = None

    exec_time = time.time() - start_time

    return {
        'identifiability': identifiability,
        'estimated_linear_DE': estimated_linear_DE,
        'exec_time': exec_time,
        'adjustment_set': adjustment_set,
        'nb_CI_tests': ci_test
    }
    

#%%
m = 10
g,y,x = random_DAG_identifiable_CDE(m, 3/m)

data = simulate_binarySCM_logistic_from_dag(g, 5000)

res = PC_CDE(data, x, y)
from pprint import pprint 
pprint(res)

print('-----')

res3 = MBbyMB_CDE(data, x, y)
pprint(res3)

print('-----')
res2 = runLocPC_CDE(data, x, y)
pprint(res2)

print('-----')
print(x,y)
print(list(g.predecessors(y)))




#%% 

##### EXPERIMENTS 
def run_identifiable_experiments(
    methods=['locpc', 'ldecc', 'pc', 'CMB', 'MBbyMB', 'MMBbyMMB'], 
    dag_sizes=[25], 
    n_experiments=50, 
    m=1, 
    n_samples=10000
):
    import numpy as np
    import pandas as pd

    summary_rows = []
    detailed_rows = []

    avg_exec_time_tracker = {method: [] for method in methods}  # Store avg times per size
    skip_methods = set()  # Methods to skip at future sizes

    for idx, size in enumerate(dag_sizes):
        prob = m / (size - 1)
        identifiable_counts = {method: 0 for method in methods}
        exec_times = {method: [] for method in methods}
        nb_CI_tests_all = {method: [] for method in methods}

        print(f"\n=== Running {n_experiments} experiments for DAG size {size} ===")
        print(f"Edge prob ≈ {prob:.3f}, Sample size = {n_samples}")

        for i in range(n_experiments):
            print(f"--- Experiment {i+1}/{n_experiments} ---")
            g, y, x = random_DAG_identifiable_CDE(size, prob)
            data = simulate_binarySCM_logistic_from_dag(g, nb_obs=n_samples)
            true_parents = list(g.predecessors(y))
            exp_id = f"{size}_{i}"

            for method in methods:
                if method in skip_methods:
                    continue

                if method == 'locpc':
                    res = runLocPC_CDE(data, x, y, linear_estimation=False, CI_test="gsq")
                elif method == 'ldecc':
                    res = ldecc_CDE(data, x, y)
                elif method == 'pc':
                    res = PC_CDE(data, x, y)
                elif method == 'CMB':
                    res = CMB_CDE(data, x, y)
                elif method == 'MBbyMB':
                    res = MBbyMB_CDE(data, x, y)
                elif method == 'MMBbyMMB':
                    res = MMBbyMMB_CDE(data, x, y)
                else:
                    continue

                ident = res.get('identifiability', False)
                exec_time = res.get('exec_time', None)
                adj_set = res.get('adjustment_set', None)
                nb_tests = res.get('nb_CI_tests', None)

                if ident:
                    identifiable_counts[method] += 1
                if exec_time is not None:
                    exec_times[method].append(exec_time)
                if nb_tests is not None:
                    nb_CI_tests_all[method].append(nb_tests)

                detailed_rows.append({
                    'experiment_id': exp_id,
                    'dag_size': size,
                    'method': method,
                    'identifiability': ident,
                    'exec_time': exec_time,
                    'adjustment_set': adj_set,
                    'true_parents': true_parents,
                    'nb_CI_tests': nb_tests
                })

        # Global summary for this size
        for method in methods:
            if method in skip_methods:
                continue

            identifiable_prop = identifiable_counts[method] / n_experiments
            avg_exec_time = np.mean(exec_times[method]) if exec_times[method] else None
            avg_nb_tests = np.mean(nb_CI_tests_all[method]) if nb_CI_tests_all[method] else None

            avg_exec_time_tracker[method].append(avg_exec_time)

            print(f"{method.upper()} → Identifiable: {identifiable_prop:.2f}, Avg CI tests: {avg_nb_tests:.1f}")

            summary_rows.append({
                'dag_size': size,
                'method': method,
                'identifiable_proportion': identifiable_prop,
                'avg_exec_time_sec': avg_exec_time,
                'avg_nb_CI_tests': avg_nb_tests
            })

            if idx > 0:
                prev_avg = avg_exec_time_tracker[method][-2]
                if prev_avg is not None and prev_avg > 20:
                    skip_methods.add(method)
                    print(f"Skipping method '{method}' for future sizes due to high exec time ({prev_avg:.2f}s at size {dag_sizes[idx-1]})")

    summary_df = pd.DataFrame(summary_rows)
    detailed_df = pd.DataFrame(detailed_rows)
    return summary_df, detailed_df


def run_non_identifiable_experiments(
    methods=['locpc', 'ldecc', 'pc', 'CMB', 'MBbyMB', 'MMBbyMMB'], 
    dag_sizes=[25], 
    n_experiments=50, 
    m=1, 
    n_samples=10000
):
    import numpy as np
    import pandas as pd

    summary_rows = []
    detailed_rows = []

    for size in dag_sizes:
        prob = m / (size - 1)
        non_identifiable_counts = {method: 0 for method in methods}
        nb_CI_tests_all = {method: [] for method in methods}

        print(f"\n=== Running {n_experiments} non-identifiable experiments for DAG size {size} ===")
        print(f"Edge prob ≈ {prob:.3f}, Sample size = {n_samples}")

        for i in range(n_experiments):
            print(f"--- Experiment {i+1}/{n_experiments} ---")
            g, y, x = random_DAG_nonidentifiable_CDE(size, prob)
            data = simulate_binarySCM_logistic_from_dag(g, nb_obs=n_samples)
            exp_id = f"{size}_{i}"

            for method in methods:
                if method == 'locpc':
                    res = runLocPC_CDE(data, x, y, linear_estimation=False, CI_test="gsq")
                elif method == 'ldecc':
                    res = ldecc_CDE(data, x, y)
                elif method == 'pc':
                    res = PC_CDE(data, x, y)
                elif method == 'CMB':
                    res = CMB_CDE(data, x, y)
                elif method == 'MBbyMB':
                    res = MBbyMB_CDE(data, x, y)
                elif method == 'MMBbyMMB':
                    res = MMBbyMMB_CDE(data, x, y)
                else:
                    continue  # unknown method

                ident = res.get('identifiability', False)
                nb_tests = res.get('nb_CI_tests', None)

                if not ident:
                    non_identifiable_counts[method] += 1
                if nb_tests is not None:
                    nb_CI_tests_all[method].append(nb_tests)

                detailed_rows.append({
                    'experiment_id': exp_id,
                    'dag_size': size,
                    'method': method,
                    'identifiability': ident,
                    'nb_CI_tests': nb_tests,
                    'discovered_hop': res.get('discovered_hop', None),
                    'NOC': res.get('NOC', None),
                    'adjustment_set': res.get('adjustment_set', None),
                })

        # Summary for this DAG size
        for method in methods:
            proportion = non_identifiable_counts[method] / n_experiments
            avg_nb_tests = np.mean(nb_CI_tests_all[method]) if nb_CI_tests_all[method] else None

            print(f"{method.upper()} → Non-identifiable: {proportion:.2f}, Avg CI tests: {avg_nb_tests:.1f}")

            summary_rows.append({
                'dag_size': size,
                'method': method,
                'non_identifiable_proportion': proportion,
                'non_identifiable_count': non_identifiable_counts[method],
                'avg_nb_CI_tests': avg_nb_tests
            })

    summary_df = pd.DataFrame(summary_rows)
    detailed_df = pd.DataFrame(detailed_rows)
    return summary_df, detailed_df



# Settings
dag_sizes = [10,20,30,40,50]
m = 2
n_samples = 5000

# Run
np.random.seed(2025)
#%%

ID_summary_df, ID_detailed_df = run_identifiable_experiments(
    methods = ['MBbyMB'],
    dag_sizes=dag_sizes,
    m=m,
    n_samples=n_samples,
    n_experiments=n_exp
)

print(ID_summary_df)


os.makedirs("BIN_output_experiments", exist_ok=True)
ID_detailed_df.to_csv("BIN_output_experiments/final_results_identifiable.csv", index=False)
ID_summary_df.to_csv("BIN_output_experiments/summary_identifiable.csv", index=False)



NONID_summary_df, NONID_detailed_df = run_non_identifiable_experiments(
    methods = ['MBbyMB'],
    dag_sizes=dag_sizes,
    m=m,
    n_samples=n_samples,
    n_experiments=n_exp
)

print(NONID_summary_df)


os.makedirs("output_experiments", exist_ok=True)

NONID_detailed_df.to_csv("BIN_output_experiments/final_results_non_identifiable.csv", index=False)
NONID_summary_df.to_csv("BIN_output_experiments/summary_nonidentifiable.csv", index=False)

# %%
# %% Add MBbyMB (rebbutal)
np.random.seed(2025)
ID_summary_df_mbbymb, ID_detailed_df_mbbymb = run_identifiable_experiments(
    methods = ["MBbyMB"],
    dag_sizes=dag_sizes,
    m=m,
    n_samples=n_samples,
    n_experiments=n_exp
)

print(ID_summary_df_mbbymb)
os.makedirs("output_MBbyMB", exist_ok=True)
ID_detailed_df_mbbymb.to_csv("output_MBbyMB/BIN_final_results_identifiable.csv", index=False)
ID_summary_df_mbbymb.to_csv("output_MBbyMB/BIN_summary_identifiable.csv", index=False)
ID_summary_df_mbbymb
NONID_summary_df_mbbymb, NONID_detailed_df_mbbymb = run_non_identifiable_experiments(
    methods = ["MBbyMB"],
    dag_sizes=dag_sizes,
    m=m,
    n_samples=n_samples,
    n_experiments=n_exp
)

print(NONID_summary_df_mbbymb)

# Save 

os.makedirs("output_MBbyMB", exist_ok=True)

NONID_detailed_df_mbbymb.to_csv("output_MBbyMB/BIN_final_results_non_identifiable.csv", index=False)
NONID_summary_df_mbbymb.to_csv("output_MBbyMB/BIN_summary_nonidentifiable.csv", index=False)


# %% BIGGER EXPERIMENTS (rebuutal)
dag_sizes = [100]
m = 2
n_samples = 5000

ID_summary_df, ID_detailed_df = run_identifiable_experiments(
    methods = ['locpc', 'ldecc', 'CMB', 'MBbyMB'],
    dag_sizes=dag_sizes,
    m=m,
    n_samples=n_samples,
    n_experiments=n_exp
)

print(ID_summary_df)


os.makedirs("output_bigDAGs_BIN", exist_ok=True)
ID_detailed_df.to_csv("output_bigDAGs_BIN/big_final_results_identifiable.csv", index=False)



NONID_summary_df, NONID_detailed_df = run_non_identifiable_experiments(
    methods = ['locpc', 'ldecc', 'CMB', 'MBbyMB'],
    dag_sizes=dag_sizes,
    m=m,
    n_samples=n_samples,
    n_experiments=n_exp
)

print(NONID_summary_df)


os.makedirs("output_bigDAGs_BIN", exist_ok=True)

NONID_detailed_df.to_csv("output_bigDAGs_BIN/big_final_results_non_identifiable.csv", index=False)

# %% REAL DATA (SACHS)
import bnlearn as bn

# Charger le dataset Sachs
data = bn.import_example(data='sachs')




print(data.head())
print(data.shape)

from causallearn.search.ConstraintBased.PC import pc

# Learn the CPDAG from data
cg = pc(data.values, alpha=0.05, indep_test='gsq')

# visualization using networkx
cg.to_nx_graph()
cg.draw_nx_graph(skel=False)


# %%
x = "Raf"
y = "Mek"

locpc_res = runLocPC_CDE(data, x, y)
pc_res = PC_CDE(data, x, y)
ldecc_res = ldecc_CDE(data, x, y)
cmb_res = CMB_CDE(data, x, y)
mbbymb_res = MBbyMB_CDE(data, x, y)
# %%
from pprint import pprint

results = {
    "LocPC": locpc_res,
    "PC": pc_res,
    "LDECC": ldecc_res,
    "CMB": cmb_res,
    "MBbyMB": mbbymb_res,
}

pprint(results)
# %%

l = LocPC(data, target_node='Mek')
# %%
l.runLocPC(1)
l.leg.draw_graph()
# %%

# %%

# %%
import pandas as pd

# URL directe du dataset (CSV brut depuis bnlearn)
df = pd.read_csv("coronary.csv", sep = ";")
print(df.head())

# %%
# Supprimer la colonne inutile
data = df.drop(columns='Unnamed: 0')

# Afficher un aperçu et la taille
print(data.head())
print(data.shape)

# Convertir les variables catégorielles en numériques (0 / 1)
# On peut utiliser map pour chaque variable si on connaît les niveaux
data['Smoking'] = data['Smoking'].map({'no': 0, 'yes': 1})
data['M. Work'] = data['M. Work'].map({'no': 0, 'yes': 1})
data['P. Work'] = data['P. Work'].map({'no': 0, 'yes': 1})
data['Pressure'] = data['Pressure'].map({'<140': 0, '>140': 1})
data['Proteins'] = data['Proteins'].map({'<3': 0, '>3': 1})
data['Family'] = data['Family'].map({'neg': 0, 'pos': 1})

# Vérifier s'il y a des valeurs manquantes
print("\nNombre de valeurs manquantes par colonne :")
print(data.isna().sum())


# %%
x = "Smoking"
y = "Proteins"

locpc_res = runLocPC_CDE(data, x, y)
pc_res = PC_CDE(data, x, y)
ldecc_res = ldecc_CDE(data, x, y)
cmb_res = CMB_CDE(data, x, y)
mbbymb_res = MBbyMB_CDE(data, x, y)
# %%
from pprint import pprint

results = {
    "LocPC": locpc_res,
    "PC": pc_res,
    "LDECC": ldecc_res,
    "CMB": cmb_res,
    "MBbyMB": mbbymb_res,
}

pprint(results)
# %%
