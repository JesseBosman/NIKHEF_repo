
"""
Goal for the optimization should be: highest neutrino efficiency at 1% muon contamination 
+ highest remaining amount of neutrinos from begin signal. So in effect, highest amount of neutrinos
at 1% muon contamination.

"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import KFold
import os
from functions import *
import optuna
import pickle

# DATAV2 DATA: MORE COMPLETE DATASET
neutr_paths = os.listdir("datav2")
muon_path = "datav2/"+neutr_paths.pop(-2)

for i, path in enumerate(neutr_paths):
    neutr_paths[i] = "datav2/"+path

neutr_types = [12,14,16,-12,-14,-16,14,-14]
is_CC = [1,0,1,1,1,1,1,0]

df_atm_muon, df_atm_neutr = load_RFD_data(muon_path, neutr_paths, neutr_types, is_CC)
# Combine the dataframes
df = pd.concat([df_atm_muon, df_atm_neutr], join= 'inner', ignore_index=True)

reco_columns = [column for column in df.columns if column[:5] == "jmuon"]
y = df.drop(columns=reco_columns)
X = df[reco_columns]

X = X.drop(columns=["jmuon_group_id"])

initial_neutr_count = y["label"].sum()

random_state = 42

# apply preselection criteria
def objective(trial):
    X_full = X.copy()
    y_full = y.copy()

    energy_crit = 20

    centre_x = X_full["jmuon_pos_x"].min() + (X_full['jmuon_pos_x'].max() - X_full['jmuon_pos_x'].min())/2
    centre_y = X_full["jmuon_pos_y"].min() + (X_full['jmuon_pos_y'].max() - X_full['jmuon_pos_y'].min())/2
    approx_r = ((X_full["jmuon_pos_x"].max() - X_full["jmuon_pos_x"].min())/2+(X_full["jmuon_pos_y"].max() - X_full['jmuon_pos_y'].min())/2)/2

    radius_crit = trial.suggest_float("radius_crit", 0.1, 1.1)
    likelihood_crit = trial.suggest_int("likelihood_crit", 0, 50)
    pos_z_crit = trial.suggest_int("pos_z_crit", 100, 250)

    threshold_likelihood = np.percentile(X_full['jmuon_likelihood'], likelihood_crit)

    mask_z_dir = X_full['jmuon_dir_z'] > 0
    mask_energy = X_full['jmuon_E'] < energy_crit
    mask_radius = np.sqrt((X_full['jmuon_pos_x'] - centre_x)**2 + (X_full['jmuon_pos_y'] - centre_y)**2) < approx_r*radius_crit
    mask_pos_z = X_full['jmuon_pos_z'] > pos_z_crit
    mask_likelihood = X_full['jmuon_likelihood'] > threshold_likelihood
    mask_geometry = mask_pos_z | mask_radius

    mask = mask_likelihood & mask_energy & mask_geometry & mask_z_dir

    X_full = X_full[mask]
    y_full = y_full[mask]

    neutrino_count_post_selection = y_full["label"].sum()

    if len(X_full) < 5:
        print("not enough samples left to split")
        return 0

    kfold = KFold(n_splits=5, shuffle=True, random_state=random_state)
    exp_data = pd.DataFrame()

    # rev_ordered_features = ['jmuon_JSTART_NPE_MIP', 'jmuon_JGANDALF_BETA0_RAD', 'jmuon_AASHOWERFIT_ENERGY', 'jmuon_JGANDALF_BETA1_RAD', 'jmuon_AASHOWERFIT_NUMBER_OF_HITS', 'jmuon_JGANDALF_LAMBDA', 'jmuon_JENERGY_NUMBER_OF_HITS', 'jmuon_t', 'jmuon_dir_x', 'jmuon_dir_y', 'jmuon_JENERGY_MAXIMAL_ENERGY', 'jmuon_JGANDALF_NUMBER_OF_ITERATIONS', 'jmuon_JENERGY_NDF', 'jmuon_JENERGY_NOISE_LIKELIHOOD', 'jmuon_JSHOWERFIT_ENERGY', 'jmuon_JENERGY_ENERGY', 'jmuon_JENERGY_MUON_RANGE_METRES', 'jmuon_E', 'jmuon_JGANDALF_NUMBER_OF_HITS', 'jmuon_JENERGY_CHI2', 'jmuon_JGANDALF_CHI2', 'jmuon_dir_z', 'jmuon_likelihood', 'jmuon_pos_z', 'jmuon_pos_y', 'jmuon_JENERGY_MINIMAL_ENERGY', 'jmuon_JSTART_NPE_MIP_TOTAL', 'jmuon_pos_x', 'jmuon_JSTART_LENGTH_METRES']

    # n_features = 14
    # X_full = X_full[rev_ordered_features[-n_features:]]

    for train_index, test_index in kfold.split(X_full):

        X_train, X_test = X_full.iloc[train_index].copy(), X_full.iloc[test_index].copy()
        y_train_compl, y_test_compl = y_full.iloc[train_index].copy(), y_full.iloc[test_index].copy()

        y_train = y_train_compl['label']

        clf = RFC(random_state=random_state, n_jobs = -1)       
        clf.fit(X_train, y_train)

        prediction_probabilities = clf.predict_proba(X_test)
        y_pred = np.argmax(prediction_probabilities, axis = 1)

        y_test_compl['muon_score'] = prediction_probabilities[:,0]
        y_test_compl['prediction'] = y_pred
        y_test_compl['jmuon_E'] = X_test['jmuon_E']

        exp_data = pd.concat([exp_data, y_test_compl])

    # Calculate the neutrino efficiency at 1% muon contamination
    muons = exp_data[exp_data['label'] == 0]
    neutrs = exp_data[exp_data['label'] == 1]


    if len(neutrs) == 0:
        print("no neutrinos were left in the sample")
        return 0

    elif len(muons) == 0:
        print("no muons were left in the sample")
        return neutrino_count_post_selection/initial_neutr_count *100

    else:

        muon_contamination_percs = muon_contamination(muons, neutrs)
        neutrino_efficiency_percs = neutrino_efficiency(neutrs)

        try:
            arg = np.argwhere(muon_contamination_percs <= 1)[-1]
        except:
            print("amount of points for lower than 1% muon contamination is ", np.sum(muon_contamination_percs <= 1))
            arg = 0

        neutrino_efficiency_at_1 = neutrino_efficiency_percs[arg]

        # Calculate the total neutrino efficiency
        total_neutrino_efficiency = neutrino_count_post_selection/initial_neutr_count * neutrino_efficiency_at_1

        return total_neutrino_efficiency

# with open("RDF_optuna_study_2.pkl", "rb") as f:
#     study = pickle.load(f)

study_name = "RDF_optuna_study_preselec"  
storage_name = "sqlite:///{}.db".format(study_name)

study = optuna.create_study(study_name = study_name, direction="maximize", storage= storage_name, load_if_exists= True)
study.optimize(objective, n_trials=1000)

print(study.best_params)
print("best total neutrino efficiency = ", study.best_value)

