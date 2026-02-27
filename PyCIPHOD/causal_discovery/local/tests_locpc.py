#%%
from causal_discovery.local.locpc import LocPC
from utils.independence_tests.basic import FisherZ, Gsq
from utils.background_knowledge.background_knowledge import BackgroundKnowledge
import numpy as np
import pandas as pd

bk = BackgroundKnowledge()
bk.add_mandatory_orientation('W', 'X')

nodes = ["X", "Y", "Z", "W"]
n = 10000

# --- Continuous linear SCM for FisherZ ---
B = np.zeros((4, 4))
B[1, 0] = 0.8; B[1, 2] = 0.5; B[1, 3] = 1.2
B[3, 2] = 1.5; B[0, 3] = 1.5
epsilon = np.random.randn(n, 4)
df_cont = pd.DataFrame(epsilon @ np.linalg.inv(np.eye(4) - B).T, columns=nodes)

# Run PC with FisherZ
locpc = LocPC(data = df_cont, 
              target_node="Y")

# %%
locpc.run(0)
# %%
locpc.leg.draw_graph({}, {})
# %%
