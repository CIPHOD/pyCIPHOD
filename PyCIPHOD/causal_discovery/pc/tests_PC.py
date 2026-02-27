#%%
from pc import PC
from utils.independence_tests.basic import FisherZ, Gsq
import numpy as np
import pandas as pd


nodes = ["X", "Y", "Z", "W"]
n = 1000

# --- Continuous linear SCM for FisherZ ---
B = np.zeros((4, 4))
B[1, 0] = 0.8; B[1, 2] = 0.5; B[1, 3] = 1.2
B[3, 2] = 1.5; B[0, 3] = 1.5
epsilon = np.random.randn(n, 4)
df_cont = pd.DataFrame(epsilon @ np.linalg.inv(np.eye(4) - B).T, columns=nodes)

# Run PC with FisherZ
pc_cont = PC(df_cont, 0.05, ci_test=FisherZ)
pc_cont.run()
pc_cont._cpdag.draw_graph({}, {})

# --- Discrete binary SCM for G² ---
# Exogenous variable
W = np.random.choice([0, 1], size=n, p=[0.4, 0.6])

# X depends on W
prob_X = np.clip(0.3 + 0.4*W, 0, 0.6)
X = np.random.binomial(1, prob_X)

# Z depends on W
prob_Z = np.clip(0.2 + 0.5*W, 0, 0.8)
Z = np.random.binomial(1, prob_Z)

# Y depends on X, Z, W
prob_Y = np.clip(0.2 + 0.3*X + 0.3*Z + 0.2*W, 0, 0.7)
Y = np.random.binomial(1, prob_Y)

df_disc = pd.DataFrame({"X": X, "Y": Y, "Z": Z, "W": W})

# Run PC with G²
pc_disc = PC(df_disc, 0.05, ci_test=Gsq)
pc_disc.run()
pc_disc._cpdag.draw_graph({}, {})
# %%