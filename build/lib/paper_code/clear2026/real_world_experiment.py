import os
import sys
from pathlib import Path
import random

import numpy as np
import pandas as pd
import networkx as nx
from pprint import pprint

# Reproducibility
SEED = 2025
random.seed(SEED)
np.random.seed(SEED)

# Add project paths
root = Path(__file__).resolve().parent
sys.path.extend([
    str(root),
    str(root.parents[1] / "pyciphod")
])

# Import CDE methods
from experiments_gaussian_scm import PC_CDE, ldecc_CDE, MBbyMB_CDE, CMB_CDE, locpc_CDE

# Load dataset
df = pd.read_csv('reproducibility/clear2026/SNDS_agreg.csv', sep=';')

# Filter out sex = 9
df_filtered = df[df['sexe'] == 9]

# Pivot to get departments vs. pathologies
data = df_filtered.pivot_table(
    index='dept',
    columns='patho_niv1',
    values='prev',
    fill_value=0
).reset_index()

# Drop non-pathology column and 'dept'
drop_cols = ["dept",
             'Pas de pathologie repérée, traitement, maternité, hospitalisation ou traitement antalgique ou anti-inflammatoire']
data = data.drop(columns=drop_cols)

# Convert to numeric
data = data.apply(pd.to_numeric, errors='coerce')

# Dataset overview
print("Dataset shape:", data.shape)
print("\nColumn types:")
print(data.dtypes)
print("\nBasic statistics:")
print(data.describe())

# Optional transformation
data_arcsin = np.arcsin(np.sqrt(data / 100))

# Define target and predictor
x = 'Insuffisance rénale chronique terminale'
y = 'Diabète'

# Run CDE methods
results = {
    "LocPC": locpc_CDE(data, x, y),
    "PC": PC_CDE(data, x, y),
    "LDECC": ldecc_CDE(data, x, y),
    "CMB": CMB_CDE(data, x, y),
    "MBbyMB": MBbyMB_CDE(data, x, y),
}

# Display results
print("\nCDE estimation results:")
pprint(results)