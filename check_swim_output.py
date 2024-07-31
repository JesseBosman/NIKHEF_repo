import uproot
import numpy as np
import pandas as pd

tree = uproot.open("/home/jbosman/ORCA10/trees/SelectedEventsTree_PID_KM3NeT_00000100_bestQ_000XXXXX.mc.gsg_neutrinos.jterbr.jppmuon_jppshower-upgoing_static.offline.dst.v9.0.root.root")["sel"]
print(tree.keys())
df = pd.DataFrame(tree.arrays(library="np"))
print(df.info())
print(df.sel_HP_track[:5])

