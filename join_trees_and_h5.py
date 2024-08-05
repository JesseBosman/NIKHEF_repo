import pandas as pd
import uproot
import numpy as np
import os
from tqdm import tqdm
import argparse



def join_trees_and_h5(det, pid_filename):
    if det == "ORCA6":
        tree_name = "scores"
    elif det == "ORCA10":
        tree_name = "sel"

    data_dir = "/data/antares/users/jwbosman/{}/pid_output/".format(det)
    trees_dir = "/data/antares/users/jwbosman/{}/trees/".format(det)

    df_h5 = pd.read_hdf(data_dir+pid_filename)
    found_match = False
    for tree_path in os.listdir(trees_dir):
        if pid_filename[:-3] in tree_path:
            found_match = True
            print("matching tree file for "+pid_filename)
            tree = uproot.open(trees_dir+tree_path)[tree_name]
            df_tree = pd.DataFrame(tree.arrays(library="np"))

    if not found_match:
        print("no matching tree file for "+ pid_filename)


    df_joined = pd.merge(df_h5, df_tree, "inner")
    df_h5 = None
    df_tree = None
    return df_joined

det = "ORCA6"
storage_dir = "/data/antares/users/jwbosman/{}/data/".format(det)
if not os.path.exists(storage_dir):
    os.makedirs(storage_dir)

for pid_filename in tqdm(os.listdir("/data/antares/users/jwbosman/{}/pid_output/".format(det))):
    df_joined = join_trees_and_h5(det, pid_filename)
    df_joined.to_hdf(path_or_buf= storage_dir+pid_filename, key="y", mode = "w")

