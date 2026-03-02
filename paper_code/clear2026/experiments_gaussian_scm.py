
#%%
# Required packages
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.linear_model import LinearRegression 
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pprint
from itertools import combinations

import random

# Set seed for reproducibility
seed = 123
random.seed(seed)
np.random.seed(seed)

# General setting : 

n_exp = 100

# Imports
from dags_generator import random_DAG_nonidentifiable_CDE, random_DAG_identifiable_CDE
from baselines.Gupta_codes.ldecc import LDECCAlgorithm
from baselines.Gupta_codes.pc_alg import PCAlgorithm
from baselines.pyCausalFS.LSL.MBs.CMB.CMB import CMB
from baselines.pyCausalFS.LSL.MBs.MBbyMB import MBbyMB
from baselines.pyCausalFS.GSL.PC import pc



import time
import numpy as np
import pandas as pd
from causallearn.utils.cit import fisherz

# Get oracle 

from itertools import combinations
    
# Simulate a linear SCM from a given DAG
def simulate_linearSCM_from_dag(dag, nb_obs=1, coef_range=(-1, 1), sigma_range=(0.5, 1)):
    """
    Simulates data from a linear Structural Causal Model (SCM) defined by a DAG with random coefficients.
    """
    v = list(nx.topological_sort(dag))
    p = len(v)

    coef = np.zeros((p, p))
    low, high = coef_range
    for j in range(p):
        pa_v = list(dag.predecessors(v[j]))
        for k in range(p):
            if v[k] in pa_v:
                while np.abs(coef[j, k]) < 0.2:
                    coef[j, k] = np.random.uniform(low, high)

    coef_inv = np.linalg.inv(np.identity(p) - coef)

    sigma_low, sigma_high = sigma_range
    sigma_vec = np.random.uniform(sigma_low, sigma_high, size=p)

    data = np.zeros((nb_obs, p))
    for i in range(nb_obs):
        eps_i = np.random.normal(loc=0, scale=sigma_vec, size=p)
        data[i, :] = np.dot(coef_inv, eps_i)

    coef_df = pd.DataFrame(coef, index=v, columns=v)
    data_df = pd.DataFrame(data, columns=v)

    return coef_df, data_df


# Algorithms
def ldecc_CDE(data, treatment, outcome):
    start_time = time.time()
    Y, X = outcome, treatment

    ldecc_alg = LDECCAlgorithm(treatment_node=Y, outcome_node=Y, use_ci_oracle=False)
    ldecc_res = ldecc_alg.run(data)

    unoriented = ldecc_res['unoriented']
    identifiability = False
    if X in ldecc_res['tmt_children'] or X not in (ldecc_res['tmt_parents'] | ldecc_res['tmt_children']) or len(unoriented) == 0 :
        identifiability = True 

    if identifiability:
        adjustment_set = list(ldecc_res['tmt_parents'])
        if X not in adjustment_set:
            estimated_linear_DE = 0
        else:
            model = LinearRegression()
            X_adjust = data[adjustment_set]
            y_target = data[Y]
            model.fit(X_adjust, y_target)
            estimated_linear_DE = model.coef_[adjustment_set.index(X)]
    else:
        estimated_linear_DE = None
        adjustment_set = None

    exec_time = time.time() - start_time

    return {
        'identifiability': identifiability,
        'estimated_linear_DE': estimated_linear_DE,
        'exec_time': exec_time,
        'adjustment_set': adjustment_set,
        'nb_CI_tests': ldecc_alg.nb_ci_tests
    }

def PC_CDE(data, treatment, outcome):
    Y, X = outcome, treatment

    pc_alg = PCAlgorithm(treatment_node=Y, outcome_node=Y, use_ci_oracle=False)
    start_time = time.time()
    pc_res_cpdag, pc_res_nb_tests = pc_alg.run(data)

    identifiability = False 
    
    if X in set(pc_res_cpdag.successors(Y)) or (X not in set(pc_res_cpdag.successors(Y)) | set(pc_res_cpdag.predecessors(Y))) or all(p not in set(pc_res_cpdag.successors(Y)) for p in pc_res_cpdag.predecessors(Y)) :
        identifiability = True
        
    if identifiability:
        adjustment_set = list(set(pc_res_cpdag.predecessors(Y)))
        if X not in adjustment_set:
            estimated_linear_DE = 0
        else:
            model = LinearRegression()
            X_adjust = data[adjustment_set]
            y_target = data[Y]
            model.fit(X_adjust, y_target)
            estimated_linear_DE = model.coef_[adjustment_set.index(X)]
    else:
        estimated_linear_DE = None
        adjustment_set = None

    exec_time = time.time() - start_time

    return {
        'identifiability': identifiability,
        'estimated_linear_DE': estimated_linear_DE,
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
    # récupérer les indices des colonnes treatment et outcome
    X = data.columns.get_loc(treatment)
    Y = data.columns.get_loc(outcome)
    start_time = time.time()

    cmb_res = CMB(data, Y, 0.05, is_discrete = False)
    
    parents, children, pc, unoriented = replace_indices_by_names(data, cmb_res)
    
    
    identifiability = False
    if (treatment in children) or (treatment not in pc) or (len(unoriented) == 0) :
        identifiability = True 

    if identifiability:
        adjustment_set = list(parents)
        if treatment not in adjustment_set:
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
        'nb_CI_tests': cmb_res[4]
    }
    
def MBbyMB_CDE(data, treatment, outcome):
    # récupérer les indices des colonnes treatment et outcome
    X = data.columns.get_loc(treatment)
    Y = data.columns.get_loc(outcome)
    start_time = time.time()

    mbbymb_res = MBbyMB(data, Y, 0.05, is_discrete = False)
    
    parents, children, pc, unoriented = replace_indices_by_names(data, mbbymb_res)
    
    
    identifiability = False
    if (treatment in children) or (treatment not in pc) or (len(unoriented) == 0) :
        identifiability = True 

    if identifiability:
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
        'nb_CI_tests': mbbymb_res[4]
    }
    
    

    





#%%

m = 10
g,y,x = random_DAG_nonidentifiable_CDE(m, 3/m)

coef, data = simulate_linearSCM_from_dag(g, 5000)

res = PC_CDE(data, x, y)
from pprint import pprint 
pprint(res)

print('-----')

res3 = MBbyMB_CDE(data, x, y)
pprint(res3)

print('-----')
res2 = runLocPC_CDE(data, x, y, linear_estimation=True)
pprint(res2)

print('-----')
print(x,y)
print(list(g.predecessors(y)))

print("true coef")
print(coef[x][y])


#%%

# F1 Score metrics
def tp(lhat, ltrue): return len([i for i in lhat if i in ltrue])
def fp(lhat, ltrue): return len([i for i in lhat if i not in ltrue])
def fn(lhat, ltrue): return len([i for i in ltrue if i not in lhat])
def precision(lhat, ltrue):
    t, f = tp(lhat, ltrue), fp(lhat, ltrue)
    return 0 if t == 0 else t / (t + f)
def recall(lhat, ltrue):
    t, f = tp(lhat, ltrue), fn(lhat, ltrue)
    return 0 if t == 0 else t / (t + f)
def f1(lhat, ltrue):
    p, r = precision(lhat, ltrue), recall(lhat, ltrue)
    return 0 if (p == 0 and r == 0) else 2 * p * r / (p + r)


def run_identifiable_experiments(
    methods=['locpc', 'ldecc', 'pc', 'CMB', 'MBbyMB', 'MMBbyMMB'], 
    dag_sizes=[25], 
    n_experiments=50, 
    m=1, 
    n_samples=10000
):
    """
    Runs experiments on identifiable DAGs to evaluate causal direct effect estimation methods.
    """
    summary_rows = []
    detailed_rows = []

    for size in dag_sizes:
        prob = m / (size - 1)
        rmse = {method: [] for method in methods}
        identifiable_counts = {method: 0 for method in methods}
        exec_times = {method: [] for method in methods}
        nb_CI_tests_all = {method: [] for method in methods}

        print(f"\n=== Running {n_experiments} experiments for DAG size {size} ===")
        print(f"Edge prob ≈ {prob:.3f}, Sample size = {n_samples}")

        for i in range(n_experiments):
            print(f"--- Experiment {i+1}/{n_experiments} ---")
            g, y, x = random_DAG_identifiable_CDE(size, prob)
            coef, data = simulate_linearSCM_from_dag(g, n_samples, sigma_range=(0.8, 1))
            true_coef = coef[x][y]
            true_parents = list(g.predecessors(y))
            exp_id = f"{size}_{i}"

            for method in methods:
                if method == 'locpc':
                    res = runLocPC_CDE(data, x, y, linear_estimation=True)
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
                est = res.get('estimated_linear_DE', None)
                exec_time = res.get('exec_time', None)
                adj_set = res.get('adjustment_set', None)
                nb_tests = res.get('nb_CI_tests', None)

                if ident:
                    identifiable_counts[method] += 1
                    if est is not None:
                        rmse[method].append((est - true_coef) ** 2)
                    if exec_time is not None:
                        exec_times[method].append(exec_time)
                    if nb_tests is not None:
                        nb_CI_tests_all[method].append(nb_tests)

                detailed_rows.append({
                    'experiment_id': exp_id,
                    'dag_size': size,
                    'method': method,
                    'identifiability': ident,
                    'estimated_linear_DE': est,
                    'true_linear_DE': true_coef,
                    'exec_time': exec_time,
                    'adjustment_set': adj_set,
                    'true_parents': true_parents,
                    'nb_CI_tests': nb_tests
                })

        # Summary for this DAG size
        for method in methods:
            rmse_final = np.sqrt(np.mean(rmse[method])) if rmse[method] else None
            identifiable_prop = identifiable_counts[method] / n_experiments
            avg_exec_time = np.mean(exec_times[method]) if exec_times[method] else None
            avg_nb_tests = np.mean(nb_CI_tests_all[method]) if nb_CI_tests_all[method] else None

            print(f"{method.upper()} → Identifiable: {identifiable_prop:.2f}, Avg CI tests: {avg_nb_tests}")

            summary_rows.append({
                'dag_size': size,
                'method': method,
                'rmse': rmse_final,
                'identifiable_proportion': identifiable_prop,
                'avg_exec_time_sec': avg_exec_time,
                'avg_nb_CI_tests': avg_nb_tests
            })

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
    """
    Runs experiments on non-identifiable DAGs to evaluate causal direct effect estimation methods.
    """
    summary_rows = []
    detailed_rows = []

    for size in dag_sizes:
        prob = m / (size - 1)
        rmse = {method: [] for method in methods}
        identifiable_counts = {method: 0 for method in methods}
        exec_times = {method: [] for method in methods}
        nb_CI_tests_all = {method: [] for method in methods}

        print(f"\n=== Running {n_experiments} non-identifiable experiments for DAG size {size} ===")
        print(f"Edge prob ≈ {prob:.3f}, Sample size = {n_samples}")

        for i in range(n_experiments):
            print(f"--- Experiment {i+1}/{n_experiments} ---")
            g, y, x = random_DAG_nonidentifiable_CDE(size, prob)
            coef, data = simulate_linearSCM_from_dag(g, n_samples, sigma_range=(0.8, 1))
            true_coef = coef[x][y]
            true_parents = list(g.predecessors(y))
            exp_id = f"{size}_{i}"

            for method in methods:
                if method == 'locpc':
                    res = runLocPC_CDE(data, x, y, linear_estimation=True)
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
                est = res.get('estimated_linear_DE', None)
                exec_time = res.get('exec_time', None)
                adj_set = res.get('adjustment_set', None)
                nb_tests = res.get('nb_CI_tests', None)

                if not ident:
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
                    'estimated_linear_DE': est,
                    'true_linear_DE': true_coef,
                    'exec_time': exec_time,
                    'adjustment_set': adj_set,
                    'true_parents': true_parents,
                    'nb_CI_tests': nb_tests
                })

        for method in methods:
            rmse_final = np.sqrt(np.mean(rmse[method])) if rmse[method] else None
            identifiable_prop = identifiable_counts[method] / n_experiments
            avg_exec_time = np.mean(exec_times[method]) if exec_times[method] else None
            avg_nb_tests = np.mean(nb_CI_tests_all[method]) if nb_CI_tests_all[method] else None

            print(f"{method.upper()} → Identifiable: {identifiable_prop:.2f}, Avg CI tests: {avg_nb_tests}")

            summary_rows.append({
                'dag_size': size,
                'method': method,
                'rmse': rmse_final,
                'identifiable_proportion': identifiable_prop,
                'avg_exec_time_sec': avg_exec_time,
                'avg_nb_CI_tests': avg_nb_tests
            })

    summary_df = pd.DataFrame(summary_rows)
    detailed_df = pd.DataFrame(detailed_rows)
    return summary_df, detailed_df


#### EXPERIMENTS
#%%
# Settings
dag_sizes = [10, 20, 30, 40, 50]
m = 2
n_samples = 5000
n_exp = 100
# Run
#%%
np.random.seed(2025)

ID_summary_df, ID_detailed_df = run_identifiable_experiments(
    methods = ['locpc', 'ldecc', 'CMB', 'pc'],
    dag_sizes=dag_sizes,
    m=m,
    n_samples=n_samples,
    n_experiments=n_exp
)

print(ID_summary_df)
os.makedirs("output_experiments", exist_ok=True)
ID_detailed_df.to_csv("output_experiments/final_results_identifiable2.csv", index=False)
ID_summary_df.to_csv("output_experiments/summary_identifiable2.csv", index=False)

NONID_summary_df, NONID_detailed_df = run_non_identifiable_experiments(
    methods = ['locpc','ldecc', 'pc', 'CMB'],
    dag_sizes=dag_sizes,
    m=m,
    n_samples=n_samples,
    n_experiments=n_exp
)

print(NONID_summary_df)

# Save 

os.makedirs("output_experiments", exist_ok=True)

NONID_detailed_df.to_csv("output_experiments/final_results_non_identifiable2.csv", index=False)
NONID_summary_df.to_csv("output_experiments/summary_nonidentifiable2.csv", index=False)


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
ID_detailed_df_mbbymb.to_csv("output_MBbyMB/final_results_identifiable.csv", index=False)
ID_summary_df_mbbymb.to_csv("output_MBbyMB/summary_identifiable.csv", index=False)
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

NONID_detailed_df_mbbymb.to_csv("output_MBbyMB/final_results_non_identifiable.csv", index=False)
NONID_summary_df_mbbymb.to_csv("output_MBbyMB/summary_nonidentifiable.csv", index=False)


#%% BIGGER EXPERIMENTS (rebbutal)

# Settings
dag_sizes = [100]
m = 2
n_samples = 5000
n_exp = 100
# Run

np.random.seed(2025)

ID_summary_df, ID_detailed_df = run_identifiable_experiments(
    methods = ['CMB', 'MBbyMB'],
    dag_sizes=dag_sizes,
    m=m,
    n_samples=n_samples,
    n_experiments=n_exp
)

print(ID_summary_df)
os.makedirs("output_bigDAGs", exist_ok=True)
ID_detailed_df.to_csv("output_bigDAGs/bigDAGs_linear_identifiable_other.csv", index=False)

NONID_summary_df, NONID_detailed_df = run_non_identifiable_experiments(
    methods = ['CMB','MBbyMB'],
    dag_sizes=dag_sizes,
    m=m,
    n_samples=n_samples,
    n_experiments=n_exp
)

print(NONID_summary_df)

# Save 

os.makedirs("output_bigDAGs", exist_ok=True)

NONID_detailed_df.to_csv("output_bigDAGs/bigDAGs_linear_nonidentifiable_other.csv", index=False)


# %% TEST PC

m = 10
g,y,x = random_DAG_identifiable_CDE(m, 2.5/m)

coef, data = simulate_linearSCM_from_dag(g, 10000)

res_pc,_ = pc(data.to_numpy(), alpha = 0.05)


data

res_pc

ldecc_CDE(data, x, y)

PC_CDE(data,x,y)
# %%
res_pc

col_index = data.columns.get_loc(y)

# indices où res_pc[x, y] == -1
parents = list(data.columns[np.where(res_pc[:, col_index] == -1)[0]])

# indices où res_pc[z, y] == 1
undirected = list(data.columns[np.where(res_pc[:, col_index] == 1)[0]])

print(res_pc)
print(parents)
print(undirected)

# %%
PC_CDE(data,x,y)
# %% REAL DATA

# Charger le CSV
df = pd.read_csv('SNDS_agreg.csv', sep=';')

# Filtrer sexe = 9
df_sexe9 = df[df['sexe'] == 9]

# Pivot table
data = df_sexe9.pivot_table(
    index='dept',
    columns='patho_niv1',
    values='prev',
    fill_value=0
).reset_index()

# Mettre toutes les colonnes sauf 'dept' au format numérique
data = data.drop(columns=["dept",
                          'Pas de pathologie repérée, traitement, maternité, hospitalisation ou traitement antalgique ou anti-inflammatoire'])
data = data.apply(pd.to_numeric, errors='coerce')

# Vérifier
print(data.head())
print("\nTypes des colonnes :")
print(data.dtypes)


data_arcsin = np.arcsin(np.sqrt(data/100))

# %%
x = 'Insuffisance rénale chronique terminale'
y = 'Diabète'

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
pc_alg = PCAlgorithm(treatment_node="Insuffisance rénale chronique terminale", outcome_node="Insuffisance rénale chronique terminale", use_ci_oracle=False)

pc_res_cpdag, pc_res_nb_tests = pc_alg.run(data)
# %%
nx.draw(pc_res_cpdag, with_labels=True, pos = nx.circular_layout(pc_res_cpdag))
# %%
