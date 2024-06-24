import ROOT
import pandas as pd
import os 
ROOT.gSystem.Load('/home/jbosman/OscProb/lib/libOscProb.so')


neutr_paths = os.listdir("datav2")
muon_path = "datav2/"+neutr_paths.pop(-2)
neutr_types = [12,14,16,-12,-14,-16,14,-14]
is_CC = [1,0,1,1,1,1,1,0]
for i, path in enumerate(neutr_paths):

    neutr_paths[i] = "datav2/"+path

df_atm_neutr = pd.DataFrame()

for i, path in enumerate(neutr_paths):
    df = pd.read_hdf(path, "y")
    df['type'] = neutr_types[i]
    df["is_CC"] = is_CC[i]
    df_atm_neutr = pd.concat([df_atm_neutr, df])

w2_name = "weight"
Ngen_name = "N_gen"
exposure_name = "exposure"


