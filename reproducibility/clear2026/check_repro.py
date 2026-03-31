from pathlib import Path
import os
import random
import numpy as np
import pandas as pd

# Set seeds in-process (good but PYTHONHASHSEED must be set externally for dict/set determinism)
SEED = 2025
random.seed(SEED)
np.random.seed(SEED)

from experiments_gaussian_scm import PC_CDE, locpc_CDE

# Load data (same processing as real_world_experiment)
root = Path(__file__).resolve().parent
data_path = root.parent.parent / 'datasets' / 'SNDS_agreg.csv'
if not data_path.exists():
    data_path = Path(__file__).resolve().parent / '../../datasets/SNDS_agreg.csv'

print('Using data file:', data_path)

df = pd.read_csv(data_path, sep=';')
# same filtering & pivot
df_filtered = df[df['sexe'] == 9]

data = df_filtered.pivot_table(
    index='dept',
    columns='patho_niv1',
    values='prev',
    fill_value=0
).reset_index()

drop_cols = ["dept",
             'Pas de pathologie repérée, traitement, maternité, hospitalisation ou traitement antalgique ou anti-inflammatoire']
try:
    data = data.drop(columns=drop_cols)
except Exception:
    pass

data = data.apply(pd.to_numeric, errors='coerce')

x = 'Insuffisance rénale chronique terminale'
y = 'Diabète'

# Run PC_CDE twice
print('\n--- PC_CDE runs ---')
res1 = PC_CDE(data, x, y)
print('First run:', res1)
res2 = PC_CDE(data, x, y)
print('Second run:', res2)

# Run locpc_CDE twice
print('\n--- locpc_CDE runs ---')
res3 = locpc_CDE(data, x, y)
print('First run locpc:', res3)
res4 = locpc_CDE(data, x, y)
print('Second run locpc:', res4)

# compare nb_CI_tests if present
for i, j, name in [(res1, res2, 'PC_CDE'), (res3, res4, 'locpc_CDE')]:
    a = i.get('nb_CI_tests')
    b = j.get('nb_CI_tests')
    print(f"{name} nb_CI_tests: run1={a}, run2={b}, equal={a==b}")
