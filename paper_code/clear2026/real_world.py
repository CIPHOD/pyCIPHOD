import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

import random

from experiments_gaussian_scm import PC_CDE, ldecc_CDE, MBbyMB_CDE, CMB_CDE, locpc_CDE

root = Path(__file__).resolve().parent
sys.path.extend([
    str(root),
    str(root.parents[1] / "PyCIPHOD")
])


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

locpc_res = locpc_CDE(data, x, y)
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

