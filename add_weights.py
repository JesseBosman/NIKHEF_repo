import ROOT
import pandas as pd
import os 
import numpy as np

ROOT.gSystem.Load('/home/jbosman/OscProb/lib/libOscProb.so')

neutr_paths = os.listdir("/home/jbosman/data_classification/")
muon_path = "/home/jbosman/data_classification/"+neutr_paths.pop(-2)
neutr_types = [12,14,16,-12,-14,-16,14,-14]
is_CC = [1,0,1,1,1,1,1,0]

df_atm_neutr = pd.DataFrame()
p = ROOT.OscProb.PMNS_Fast()

w2_name = "weight"
Ngen_name = "N_gen"
exposure_name = "exposure"

prem_model = ROOT.PremModel()

# need to find out what the fluxes are for the different neutrino types.
flux_nu_mu = None
flux_nu_e = None
flux_nu_tau = None

# ORCA is roughly at 42°45'20"N 6°05'04"E, however this not used by prem_model and not exact.
# default r = 6368 km which is at bottom of ocean layer.

for i, path in enumerate(neutr_paths):
    path = "/home/jbosman/data_classification/"+path
    df = pd.read_hdf(path, "y")
    df['type'] = neutr_types[i]
    df["is_CC"] = is_CC[i]

    # to use the functions we need to iterate through each point seperately.
    for index, row in df.iterrows():

        # need to calculate path through earth.
        cos_theta = np.sqrt(1 - row["dir_z"]**2)
        prem_model.FillPath(cos_theta)
        p.SetPath(prem_model.GetNuPath())

        # now calculate the oscillation probabilities
        energy = row["energy"]
        particle_type = row["type"]
        if particle_type == 12:
            p.SetIsNuBar(False)
            flux_oscillation = flux_nu_e*p.Prob(0, 0, energy) + flux_nu_mu*p.Prob(1, 0, energy) + flux_nu_tau*p.Prob(2, 0, energy)
        elif particle_type == -12:
            p.SetIsNuBar(True)
            flux_oscillation = flux_nu_e*p.Prob(0, 0, energy) + flux_nu_mu*p.Prob(1, 0, energy) + flux_nu_tau*p.Prob(2, 0, energy)
        
        elif particle_type == 14:
            p.SetIsNuBar(False)
            flux_oscillation = flux_nu_e*p.Prob(0, 1, energy) + flux_nu_mu*p.Prob(1, 1, energy) + flux_nu_tau*p.Prob(2, 1, energy)

        elif particle_type == -14:
            p.SetIsNuBar(True)
            flux_oscillation = flux_nu_e*p.Prob(0, 1, energy) + flux_nu_mu*p.Prob(1, 1, energy) + flux_nu_tau*p.Prob(2, 1, energy)
        
        elif particle_type == 16:
            p.SetIsNuBar(False)
            flux_oscillation = flux_nu_e*p.Prob(0, 2, energy) + flux_nu_mu*p.Prob(1, 2, energy) + flux_nu_tau*p.Prob(2, 2, energy)
        
        elif particle_type == -16:
            p.SetIsNuBar(True)
            flux_oscillation = flux_nu_e*p.Prob(0, 2, energy) + flux_nu_mu*p.Prob(1, 2, energy) + flux_nu_tau*p.Prob(2, 2, energy)

        else:
            raise ValueError("Particle type not recognized")
        

        row["flux_weight"] = row[w2_name]/row[Ngen_name]*row[exposure_name]*flux_oscillation

        df.loc[index] = row

    df.to_hdf(path, "y", mode="w")



 