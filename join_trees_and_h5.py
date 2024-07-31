import pandas as pd
import uproot
import numpy as np
import os



def join_trees_and_h5(tree_path, h5_path, output_path):
    tree = uproot.open(tree_path)["sel"]
    df_tree = pd.DataFrame(tree.arrays(library="np"))

    df_h5 = pd.read_hdf(h5_path)

    print(len(df_tree), len(df_h5["muonscore"]))