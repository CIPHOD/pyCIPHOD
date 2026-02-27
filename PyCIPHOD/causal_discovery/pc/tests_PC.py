#%%
from pc import PC
from utils.independence_tests.basic import CiTests, FisherZ

import numpy as np
import pandas as pd

np.random.seed(42)
nodes = ["X", "Y", "Z", "W"]
B = np.zeros((4, 4))
B[1, 0] = 0.8
B[1, 2] = 0.5
B[3, 1] = 1.2
epsilon = np.random.randn(1000, 4)
X_values = epsilon @ np.linalg.inv(np.eye(4) - B).T
df = pd.DataFrame(X_values, columns=nodes)
print(df.head())

# %%
pc = PC(df, 0.05, ci_test = FisherZ)
pc._skeleton()
# %%
