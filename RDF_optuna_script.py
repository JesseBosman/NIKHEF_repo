
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
neutr_types = ["neutrino_elec-CC","neutrino_muon-NC","neutrino_tau-CC","neutrino_anti-elec-CC","neutrino_anti-muon-CC","neutrino_anti-tau-CC","neutrino_muon-CC","neutrino_anti-muon-NC"]

df_atm_muon, df_atm_neutr = load_RFD_data(muon_path, neutr_paths, neutr_types)
# Combine the dataframes
df = pd.concat([df_atm_muon, df_atm_neutr], join= 'inner', ignore_index=True)

reco_columns = [column for column in df.columns if column[:5] == "jmuon"]
y = df.drop(columns=reco_columns)
X = df[reco_columns]

X = X.drop(columns=["jmuon_group_id"])

initial_neutr_count = y["label"].sum()

# apply preselection criteria
def objective(trial):
    X_full = X.copy()
    y_full = y.copy()

    energy_crit = 20

    centre_x = X_full["jmuon_pos_x"].min() + (X_full['jmuon_pos_x'].max() - X_full['jmuon_pos_x'].min())/2
    centre_y = X_full["jmuon_pos_y"].min() + (X_full['jmuon_pos_y'].max() - X_full['jmuon_pos_y'].min())/2
    approx_r = ((X_full["jmuon_pos_x"].max() - X_full["jmuon_pos_x"].min())/2+(X_full["jmuon_pos_y"].max() - X_full['jmuon_pos_y'].min())/2)/2

    radius_crit = trial.suggest_float("radius_crit", 0.1, 1.5)
    likelihood_crit = trial.suggest_float("likelihood_crit", 0, 99)
    z_pos_crit_upper = trial.suggest_float("z_pos_crit_upper", 128, 260)
    z_pos_crit_lower = trial.suggest_float("z_pos_crit_lower", 0, 132)
    z_dir_crit_upper = trial.suggest_float("z_dir_crit_upper", -1, 1)
    z_dir_crit_lower = trial.suggest_float("z_dir_crit_lower", -1, 1)

    max_samples = trial.suggest_float("max_samples", 0.1, 0.95)
    max_features = trial.suggest_float("max_features", 0.1, 0.95)
    n_estimators = trial.suggest_int("n_estimators", 101, 501)
    class_weight = trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample", None])
    criterion = trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"])
    min_samples_split = trial.suggest_float("min_samples_split", low = 0.00001, high = 0.01)

    threshold_likelihood = np.percentile(X_full['jmuon_likelihood'], likelihood_crit)
    mask_pos_dir_z_upper = (X_full['jmuon_pos_z'] < z_pos_crit_upper)|(X_full['jmuon_dir_z'] > z_dir_crit_upper)
    mask_pos_dir_z_lower = (X_full['jmuon_pos_z'] > z_pos_crit_lower)|(X_full['jmuon_dir_z'] > z_dir_crit_lower)
    mask_energy = X_full['jmuon_E'] < energy_crit
    mask_radius = np.sqrt((X_full['jmuon_pos_x'] - centre_x)**2 + (X_full['jmuon_pos_y'] - centre_y)**2) < approx_r*radius_crit
    mask_likelihood = X_full['jmuon_likelihood'] > threshold_likelihood

    mask = mask_pos_dir_z_upper & mask_pos_dir_z_lower  & mask_likelihood & mask_energy & mask_radius

    X_full = X_full[mask]
    y_full = y_full[mask]

    neutrino_count_post_selection = y_full["label"].sum()

    if len(X_full) < 5:
        print("not enough samples left to split")
        return 0

    kfold = KFold(n_splits=5, shuffle=True)
    exp_data = pd.DataFrame()

    rev_ordered_features = ['jmuon_JSTART_NPE_MIP', 'jmuon_JGANDALF_BETA0_RAD', 'jmuon_AASHOWERFIT_ENERGY', 'jmuon_JGANDALF_BETA1_RAD', 'jmuon_AASHOWERFIT_NUMBER_OF_HITS', 'jmuon_JGANDALF_LAMBDA', 'jmuon_JENERGY_NUMBER_OF_HITS', 'jmuon_t', 'jmuon_dir_x', 'jmuon_dir_y', 'jmuon_JENERGY_MAXIMAL_ENERGY', 'jmuon_JGANDALF_NUMBER_OF_ITERATIONS', 'jmuon_JENERGY_NDF', 'jmuon_JENERGY_NOISE_LIKELIHOOD', 'jmuon_JSHOWERFIT_ENERGY', 'jmuon_JENERGY_ENERGY', 'jmuon_JENERGY_MUON_RANGE_METRES', 'jmuon_E', 'jmuon_JGANDALF_NUMBER_OF_HITS', 'jmuon_JENERGY_CHI2', 'jmuon_JGANDALF_CHI2', 'jmuon_dir_z', 'jmuon_likelihood', 'jmuon_pos_z', 'jmuon_pos_y', 'jmuon_JENERGY_MINIMAL_ENERGY', 'jmuon_JSTART_NPE_MIP_TOTAL', 'jmuon_pos_x', 'jmuon_JSTART_LENGTH_METRES']

    n_features = 14
    X_selected_features = X_full[rev_ordered_features[-n_features:]]

    for train_index, test_index in kfold.split(X_selected_features):

        X_train, X_test = X_selected_features.iloc[train_index].copy(), X_selected_features.iloc[test_index].copy()
        y_train_compl, y_test_compl = y_full.iloc[train_index].copy(), y_full.iloc[test_index].copy()

        y_train = y_train_compl['label']

        clf = RFC(n_estimators=n_estimators, criterion= criterion, random_state=42, class_weight=class_weight, oob_score=True, verbose = 0, n_jobs=-1, min_samples_split=min_samples_split, max_samples = max_samples, max_features = max_features)       
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

study_name = "RDF_optuna_study_2"  
storage_name = "sqlite:///{}.db".format(study_name)

study = optuna.create_study(study_name = study_name, direction="maximize", storage= storage_name, load_if_exists= True)
study.optimize(objective, n_trials=1000)

print(study.best_params)
print("best total neutrino efficiency = ", study.best_value)

