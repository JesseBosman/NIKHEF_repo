import pandas as pd
import numpy as np
# from functions import load_data, remove_nans
# import matplotlib.pyplot as plt
import os
from functions import load_data
from tqdm import tqdm
import pickle as pkl
from tqdm import tqdm

def create_aliases():
    with open("NIKHEF_repo/aliases.pkl", "wb") as f:
        pkl.dump({}, f)

def check_aliases(det):
    with open("NIKHEF_repo/aliases.pkl", "rb") as f:
        aliases = pkl.load(f)
    
    if det == "ORCA6":
        data = pd.concat([pd.read_hdf("/data/antares/users/jwbosman/ORCA6/data/mcv7.2.gsg_anti-elec-CC_100-500GeV.km3sim.jorcarec.jsh.aanet.dst_merged.h5"), 
                         pd.read_hdf("/data/antares/users/jwbosman/ORCA6/data/v7.2.mc.mupage_tuned.sirene.jorcarec.jsh.aanet.dst_merged.h5")])
        
    elif det == "ORCA10":
        data = pd.concat([pd.read_hdf("/data/antares/users/jwbosman/ORCA10/data/KM3NeT_00000100_bestQ_000XXXXX.mc.gsg_neutrinos.jterbr.jppmuon_jppshower-upgoing_static.offline.dst.v9.0.h5")
                            , pd.read_hdf("/data/antares/users/jwbosman/ORCA10/data/KM3NeT_00000100_bestQ_000114XX.mc.mupage.jterbr.jppmuon_jppshower-upgoing_static.offline.dst.v9.0.h5")])


    for column in tqdm(data.columns, desc = f"Looping over columns in {det} in check_aliases"):
        already_added = False

        duplicates = [col for col in data.columns if ((data[column].equals(data[col]))&(col!=column))]
        
        if (len(duplicates)!=0):
            if column in aliases.keys():
                already_added = True

            else:
                for key in aliases.keys():
                    if column in aliases[key]:
                        already_added = True
                        break

            if not already_added:
                for duplicate in duplicates:
                    if duplicate in aliases.keys():
                        new_set = set(duplicates+aliases[duplicate]+[column])
                        aliases.update({duplicate: list(new_set)})
                        print(f"{column} has duplicates: {aliases[duplicate]} and has been aliased to {duplicate}")
                        already_added = True
                        break

                    else:
                        for key in aliases.keys():
                            if duplicate in aliases[key]:
                                print(f"{column} has duplicates: {duplicates} and is aliased to {key}")
                                new_set = set(duplicates+aliases[key]+[column])
                                aliases.update({key: list(new_set)})
                                already_added = True
                                break

            if not already_added:
                print(f"{column} has duplicates: {duplicates} and has been added to the aliases file.")
                aliases[column] = duplicates+[column] 
        else:
            continue

            
    with open("NIKHEF_repo/aliases.pkl", "wb") as f:
        pkl.dump(aliases, f)

def change_preferred_alias():
    with  open("NIKHEF_repo/aliases.pkl", "rb") as f:
        aliases = pkl.load(f)

    new_aliases = {}

    for key in tqdm(aliases.keys(), desc = "Looping over keys in change_preferred_alias"):
        print("the current preferrerd alias is {} with the aliases {}".format(key, aliases[key]))
        print("If you want to change the preferred alias, type the new preferred key. If you want to keep the preferred key, press enter.")
        new_key = input()
        if new_key == "":
            new_key = key
        else:
            print("The new preferred aliases is {}".format(new_key))
        
        new_aliases[new_key] = aliases[key]

    with open("NIKHEF_repo/aliases.pkl", "wb") as f:
        pkl.dump(new_aliases, f)

def apply_preferred_alias_and_remove_duplicates():
    with open("NIKHEF_repo/aliases.pkl", "rb") as f:
        aliases = pkl.load(f)
    
    for det in ["ORCA6", "ORCA10"]:
        for file in tqdm(os.listdir("/data/antares/users/jwbosman/{}/data/".format(det)), desc=f"\n Looping over {det} files in apply_preferred_alias"):

            data = pd.read_hdf("/data/antares/users/jwbosman/{}/data/".format(det)+file)
            for column in tqdm(data.columns, desc= " \n Looping over {} columns in apply_preferred_alias".format(det)):
                keys = aliases.keys()	
                if column in aliases.keys():
                    print("\n Column {} already has the preferred name. ".format(column))
                else:
                    for key in aliases.keys():
                        test_aliases = aliases[key]
                        if column in aliases[key]:
                            print("\n Renaming column {} to {}".format(column, key))
                            data.rename(columns = {column: key}, inplace = True)
                            break

            
            print("Removing duplicates")
            print("N columns before removing duplicates: {}".format(len(data.columns)))
            data = data.loc[:,~data.columns.duplicated()]
            print("N columns after removing duplicates: {}".format(len(data.columns)))

            data.to_hdf("/data/antares/users/jwbosman/{}/data/".format(det)+file, key = "y", mode = "w")

def check_nans(data, columns_with_nans):
    for column in data.columns:
        if len(data[column].isnull())>0:
            columns_with_nans+= [column]
    return list(set(columns_with_nans))

def check_non_overlapping_columns(existing_list, new_columns, non_overlapping):
    for column in new_columns:
        if column in existing_list:
            pass
        else:
            non_overlapping+= [column]
            existing_list+= [column]
        
    return existing_list, non_overlapping

def remove_columns(data, columns_to_remove):
    for column in columns_to_remove:
        try:
            data = data.drop(column, axis = 1)
        except:
            print(f"Column {column} not in data")
    
    return data
        


if __name__ == "__main__":
    # create_aliases()
    # for det in ["ORCA6", "ORCA10"]:
    #     check_aliases(det)

    # change_preferred_alias()
    # apply_preferred_alias_and_remove_duplicates()
    with open("NIKHEF_repo/aliases.pkl", "rb") as f:
        aliases = pkl.load(f)

    print(aliases)
    # columns_with_nans = []
    # existing_list = list(pd.read_hdf("/data/antares/users/jwbosman/ORCA10/data/KM3NeT_00000100_bestQ_000XXXXX.mc.gsg_neutrinos.jterbr.jppmuon_jppshower-upgoing_static.offline.dst.v9.0.h5").columns)
    # non_overlapping = []
    # for det in ["ORCA6", "ORCA10"]:
    #     data_dir = "/data/antares/users/jwbosman/{}/data/".format(det)
    #     columns_with_nans = []
    #     existing_list = list(pd.read_hdf(data_dir+os.listdir(data_dir)[0]).columns)
    #     non_overlapping = []

    #     for path in os.listdir("/data/antares/users/jwbosman/{}/data/".format(det)):
    #         data = pd.read_hdf("/data/antares/users/jwbosman/{}/data/".format(det)+path)
    #         columns_with_nans = check_nans(data, columns_with_nans)
    #         existing_list, non_overlapping = check_non_overlapping_columns(existing_list, data.columns, non_overlapping)
        
    #     print("Det: {}".format(det))
    #     print("Columns with nans: {}".format(columns_with_nans))
    #     print("Non overlapping columns: {}".format(non_overlapping))

