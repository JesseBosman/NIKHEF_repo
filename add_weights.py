import ROOT
import pandas as pd
import os 
import numpy as np
import km3flux
from tqdm import tqdm

def return_honda_fluxes(row, p, prem_model, flux, energy):
    prem_model.FillPath(row["cos_zenith_true"])
    p.SetPath(prem_model.GetNuPath())

    

    # as the energy in the flux data is binned, we need to find out the correct energy value to take.
    # see that I do this doubly, could make this more efficient/quicker.
    arg_min = np.argwhere(flux._data.energy<=energy)[-1]
    arg_max = arg_min+1
    min_max_energy_range = np.array([flux._data.energy[arg_min], flux._data.energy[arg_max]])

    diff = np.abs(min_max_energy_range-energy)

    energy_crit_flux = min_max_energy_range[np.argmin(diff)]

    # select the correct data subset
    honda_fluxes = flux._data[(flux._data.cosz_min < row["cos_zenith_true"]) & (flux._data.cosz_max >=row["cos_zenith_true"]) & (flux._data.energy == energy_crit_flux)]

    return honda_fluxes, p

ROOT.gSystem.Load('/home/jbosman/OscProb/lib/libOscProb.so')
parent_dir = "/home/jbosman/datav7.2_flux_weighed_reduced/"
neutr_paths = os.listdir(parent_dir)
neutr_paths = [path for path in neutr_paths if "mupage" not in path]

df_atm_neutr = pd.DataFrame()
p = ROOT.OscProb.PMNS_Fast()

# exposure_name = "exposure"

prem_model = ROOT.OscProb.PremModel()
honda = km3flux.flux.Honda()
flux = honda.flux(2014, "Frejus", "azimuth", 'min')

# no values for nu_tau in flux, so set to 0.
# flux_nu_tau = 0

# ORCA is roughly at 42°45'20"N 6°05'04"E, however this not used by prem_model and not exact.
# default r = 6368 km which is at bottom of ocean layer.

for i, path in enumerate([neutr_paths[0]]):
    print(path)
    total_path = parent_dir+path
    df = pd.read_hdf(total_path)
    # df["flux_weight"] = 0.0'
    df["manual_flux_weight"] = 0.0

    particle_type = df["pdgid"].iloc[0]
    print(particle_type)

    # to use the functions we need to iterate through each point seperately.
    if particle_type == 12:
        p.SetIsNuBar(False)
        for index, row in tqdm(df.iterrows(), total = len(df)):
            energy = row["energy"]
            fluxes, p = return_honda_fluxes(row, p, prem_model, flux, energy)
            flux_nu_e = fluxes["nue"][0]
            flux_nu_mu = fluxes["numu"][0]
            flux_oscillation = flux_nu_e*p.Prob(0, 0, energy) + flux_nu_mu*p.Prob(1, 0, energy)#+ flux_nu_tau*p.Prob(2, 0, energy)
            # flux_weight = row["weight_one_year"]*flux_oscillation'
            # row["flux_weight"] = flux_weight
            flux_weight = row["w2"]/row["ngen"]*row["exposure"] *flux_oscillation
            row["manual_flux_weight"] = flux_weight
            
            df.loc[index] = row
            
    elif particle_type == -12:
        p.SetIsNuBar(True)
        for index, row in tqdm(df.iterrows(), total = len(df)):
            energy = row["energy"]
            fluxes, p = return_honda_fluxes(row, p, prem_model, flux, energy)
            flux_nu_e = fluxes["anue"][0]
            flux_nu_mu = fluxes["anumu"][0]
            flux_oscillation = flux_nu_e*p.Prob(0, 0, energy) + flux_nu_mu*p.Prob(1, 0, energy)# + flux_nu_tau*p.Prob(2, 0, energy)
            # flux_weight = row["weight_one_year"]*flux_oscillation
            # row["flux_weight"] = flux_weight
            flux_weight = row["w2"]/row["ngen"]*row["exposure"] *flux_oscillation
            row["manual_flux_weight"] = flux_weight

            df.loc[index] = row
    
    elif particle_type == 14:
        p.SetIsNuBar(False)
        for index, row in tqdm(df.iterrows(), total = len(df)):
            energy = row["energy"]
            fluxes, p = return_honda_fluxes(row, p, prem_model, flux, energy)
            flux_nu_e = fluxes["nue"][0]
            flux_nu_mu = fluxes["numu"][0]
            flux_oscillation = flux_nu_e*p.Prob(0, 1, energy) + flux_nu_mu*p.Prob(1, 1, energy)# + flux_nu_tau*p.Prob(2, 1, energy)
            # flux_weight = row["weight_one_year"]*flux_oscillation
            # row["flux_weight"] = flux_weight
            flux_weight = row["w2"]/row["ngen"]*row["exposure"] *flux_oscillation
            row["manual_flux_weight"] = flux_weight
            df.loc[index] = row

    elif particle_type == -14:
        p.SetIsNuBar(True)
        for index, row in tqdm(df.iterrows(), total = len(df)):
            energy = row["energy"]
            fluxes, p = return_honda_fluxes(row, p, prem_model, flux, energy)
            flux_nu_e = fluxes["anue"][0]
            flux_nu_mu = fluxes["anumu"][0]
            flux_oscillation = flux_nu_e*p.Prob(0, 1, energy) + flux_nu_mu*p.Prob(1, 1, energy)# + flux_nu_tau*p.Prob(2, 1, energy)
            # flux_weight = row["weight_one_year"]*flux_oscillation
            # row["flux_weight"] = flux_weight
            flux_weight = row["w2"]/row["ngen"]*row["exposure"] *flux_oscillation
            row["manual_flux_weight"] = flux_weight
            df.loc[index] = row
        
    elif particle_type == 16:
        p.SetIsNuBar(False)
        for index, row in tqdm(df.iterrows(), total = len(df)):
            energy = row["energy"]
            fluxes, p = return_honda_fluxes(row, p, prem_model, flux, energy)
            flux_nu_e = fluxes["nue"][0]
            flux_nu_mu = fluxes["numu"][0]
            flux_oscillation = flux_nu_e*p.Prob(0, 2, energy) + flux_nu_mu*p.Prob(1, 2, energy)# + flux_nu_tau*p.Prob(2, 2, energy)
            # flux_weight = row["weight_one_year"]*flux_oscillation
            # row["flux_weight"] = flux_weight
            flux_weight = row["w2"]/row["ngen"]*row["exposure"] *flux_oscillation
            row["manual_flux_weight"] = flux_weight
            df.loc[index] = row
        
    elif particle_type == -16:
        p.SetIsNuBar(True)
        for index, row in tqdm(df.iterrows(), total = len(df)):
            energy = row["energy"]
            fluxes, p = return_honda_fluxes(row, p, prem_model, flux, energy)
            flux_nu_e = fluxes["anue"][0]
            flux_nu_mu = fluxes["anumu"][0]
            flux_oscillation = flux_nu_e*p.Prob(0, 2, energy) + flux_nu_mu*p.Prob(1, 2, energy)# + flux_nu_tau*p.Prob(2, 2, energy)
            # flux_weight = row["weight_one_year"]*flux_oscillation
            # row["flux_weight"] = flux_weight
            flux_weight = row["w2"]/row["ngen"]*row["exposure"] *flux_oscillation
            row["manual_flux_weight"] = flux_weight
            df.loc[index] = row

    else:
        raise ValueError("Particle type not recognized")
        
    

    # df.to_hdf(path, "y", mode="w")
    # print(flux_weight)
    # print(df.columns)
    print(df["manual_flux_weight"][:20])

    # if not os.path.isdir("/home/jbosman/datav7.2_pid_h5_flux_weighted/"):
    #     os.mkdir("/home/jbosman/datav7.2_pid_h5_flux_weighted/")
    # df.to_hdf("/home/jbosman/datav7.2_pid_h5_flux_weighted/"+path, key = "y", mode="w")
    df.to_hdf("/home/jbosman/NIKHEF_repo/"+path, key = "y", mode="w")



 