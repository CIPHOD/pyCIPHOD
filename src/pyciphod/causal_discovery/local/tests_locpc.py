#%%
from causal_discovery.local.locpc import LocPC
from causal_discovery.pc.pc import PC
from PyCIPHOD.utils.independence_tests.basic import FisherZ, Gsq
from PyCIPHOD.utils.background_knowledge.background_knowledge import BackgroundKnowledge
import numpy as np
import pandas as pd

bk = BackgroundKnowledge()
bk.add_mandatory_orientation('W', 'X')

nodes = ["X", "Y", "Z", "W"]
n = 10000

# --- Continuous linear SCM for FisherZ ---
B = np.zeros((4, 4))
B[1, 0] = 0.8; B[1, 2] = 0.5; B[1, 3] = 1.2
B[3, 2] = 1.5; B[0, 3] = -10
epsilon = np.random.randn(n, 4)
df_cont = pd.DataFrame(epsilon @ np.linalg.inv(np.eye(4) - B).T, columns=nodes)

# Run PC with FisherZ
locpc = LocPC(data = df_cont, twd = True)

# %%
locpc.run("Y",1)
# %%
locpc.leg.draw_graph({}, {})
# %%
locpc = LocPC(data = df_cont, twd = True)

res = locpc.run_locPC_CDE('X','Y')
locpc.nb_ci_tests
# %%
pc = PC(df_cont, twd = True)
pc.run()
pc.nb_ci_tests
# %%
