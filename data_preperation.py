import pandas as pd
import numpy as np
# from functions import load_data, remove_nans
# import matplotlib.pyplot as plt
import os

# import seaborn as sns

analysis_or_saving = "not_analysis"

if analysis_or_saving == "analysis":

    df_muons = pd.read_hdf("/home/jbosman/datav7.2_pid_h5/v7.2.mc.mupage_tuned.sirene.jorcarec.jsh.aanet.dst_merged.root.h5")
    print("muon data loaded")
    df_neutrinos = pd.read_hdf("/home/jbosman/datav7.2_pid_h5/mcv7.2.gsg_anti-elec-CC_1-100GeV.km3sim.jorcarec.jsh.aanet.dst_merged.root.h5")
    print("neutrino data loaded")
    # add bjorkeny column to muon data
    df_muons["T.sum_mc_nu.by"] = -1

    # checking for non-overlapping columns

    nu_col_not_in_mu = []
    for column_nu in df_neutrinos.columns:
        is_in = False
        for column_mu in df_muons.columns:
            if column_nu == column_mu:
                is_in = True
                break
        
        if not is_in:
            nu_col_not_in_mu.append(column_nu)
            print("column ", column_nu, " not in muon data")

    mu_col_not_in_nu = []
    for column_mu in df_muons.columns:
        is_in = False
        for column_nu in df_neutrinos.columns:
            if column_nu == column_mu:
                is_in = True
                break

        if not is_in:
            mu_col_not_in_nu.append(column_mu)
            print("column ", column_mu, " not in neutrino data")
    
    print("muon columns not in neutrino data: ", mu_col_not_in_nu)
    print("neutrino columns not in muon data: ", nu_col_not_in_mu)

    # remove non-overlapping columns
    df_muons = df_muons.drop(columns=mu_col_not_in_nu)
    df_neutrinos = df_neutrinos.drop(columns=nu_col_not_in_mu)

    # check nans
    print("neutrinos"+"-"*20)
    for column in df_neutrinos.columns:
        if np.any(df_neutrinos[column].isnull()):
            n_nan = np.sum(df_muons[column].isnull())
            print("column " +column + " has " + str(n_nan) + " nans")

    print("muons"+"-"*20)
    for column in df_muons.columns:
        if np.any(df_muons[column].isnull()):
            n_nan = np.sum(df_muons[column].isnull())
            print("column " +column + " has " + str(n_nan) + " nans")

    # join data

    df = pd.concat([df_muons, df_neutrinos], ignore_index=True)
    df_muons = None
    df_neutrinos = None

    columns_to_remove = []
    # remove columns with only 1 unique value
    columns_to_remove = []
    for column in df.columns:
        if len(df[column].unique()) == 1:
            print("column {} has only 1 unique value, which is {}".format(column, df[column].unique()))
            df = df.drop(column, axis = 1)
            columns_to_remove.append(column)

    # get aliases
    aliases_tot = {}
    for column1 in df.columns:
        aliases = []
        for column2 in df.columns:
            if column1 != column2:
                if np.array_equal(df[column1], df[column2]):
                    aliases.append(column2)

        if len(aliases) > 0:
            aliases_tot[column1] = aliases
            print(f"{column1} has aliases {aliases}")
        
    # remove aliases
    already_removed = []
    for column1 in df.columns:
        # print("current column1 "+column1)
        if np.any([column1 == x for x in already_removed]):
            # print(f"{column1} is already removed")
            pass
        else:
            for column2 in df.columns:
                if column1 != column2:
                    if np.array_equal(df[column1], df[column2]):
                        print(f"{column1} is equal to {column2}")
                        df.drop(column2, axis=1, inplace=True)
                        already_removed.append(column2)
    
    columns_to_remove.extend(already_removed)

    print("after joining, removed columns: "+str(columns_to_remove))

else:
    columns_to_remove_both = ['T.sum_mc_evt.E_min_gen', "E.trks.fitinf[:,0,3]", 'run_duration', 'weight',]
    columns_to_remove_muons = ['E.mc_trks.type[:,0]', 'ratio_track_100_jg', 'ratio_casc_100_jsh', 'ratio_track_tot_jg']
    columns_to_remove_neutrinos = ['E.w2list[:,1]', 'E.w2list[:,5]', 'E.w[:,0]', 'E.w[:,1]', 'T.sum_mc_evt.n_gen']
    columns_to_remove_muons.extend(columns_to_remove_both)
    columns_to_remove_neutrinos.extend(columns_to_remove_both)

    df_muons = pd.read_hdf("/home/jbosman/datav7.2_pid_h5_flux_weighted/v7.2.mc.mupage_tuned.sirene.jorcarec.jsh.aanet.dst_merged.root.h5")
    print("muon data loaded")
    df_muons.drop(columns=columns_to_remove_muons, inplace=True)
    # df_muons["flux_weight"] = df_muons["weight_one_year"].values
    # df_muons["T.sum_mc_nu.by"] = -1
    df_muons.to_hdf("/home/jbosman/datav7.2_flux_weighed_reduced/v7.2.mc.mupage_tuned.sirene.jorcarec.jsh.aanet.dst_merged.root.h5", key='y')
    print("saved muon data"+ "-"*20)
    df_muons = None

    neutr_dir = "/home/jbosman/datav7.2_pid_h5_flux_weighted/"
    neutr_paths = os.listdir(neutr_dir)
    neutr_paths = [path for path in neutr_paths if "mupage" not in path]
    for path in neutr_paths:
        df_neutrinos = pd.read_hdf(neutr_dir+path)
        print(path+" loaded")
        df_neutrinos.drop(columns=columns_to_remove_neutrinos, inplace=True)
        df_neutrinos.to_hdf("/home/jbosman/datav7.2_flux_weighed_reduced/"+path, key='y')
        print("saved neutrino data "+path+ "-"*20)
        df_neutrinos = None
    print("done")

    