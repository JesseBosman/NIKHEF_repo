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

print("Initial neutrino count ",initial_neutr_count)

X_full = X.copy()
y_full = y.copy()

energy_crit = 20

centre_x = X_full["jmuon_pos_x"].min() + (X_full['jmuon_pos_x'].max() - X_full['jmuon_pos_x'].min())/2
centre_y = X_full["jmuon_pos_y"].min() + (X_full['jmuon_pos_y'].max() - X_full['jmuon_pos_y'].min())/2
approx_r = ((X_full["jmuon_pos_x"].max() - X_full["jmuon_pos_x"].min())/2+(X_full["jmuon_pos_y"].max() - X_full['jmuon_pos_y'].min())/2)/2

with open("RDF_optuna_study.pkl", "rb") as f:
    study = pickle.load(f)

best_params = study.best_params

dir_z_crit = best_params["dir_zit_crit"]
radius_crit = best_params["radius_crit"]
likelihood_crit = best_params["likelihood_crit"]
z_pos_dir_crit_upper = best_params["z_pos_dir_crit_upper"]
z_pos_dir_crit_lower = best_params["z_pos_dir_crit_lower"]

mask_z_dir = X_full['jmuon_dir_z'] > dir_z_crit
mask_energy = X_full['jmuon_E'] < energy_crit
mask_pos_dir_z = (X_full['jmuon_pos_z'] > (-27.9*np.square(X_full['jmuon_dir_z']) - 15.51*X_full['jmuon_dir_z'] + z_pos_dir_crit_upper)) | (X_full['jmuon_pos_z'] < (-27.9*np.square(X_full['jmuon_dir_z']) - 15.51*X_full['jmuon_dir_z'] + z_pos_dir_crit_lower))
mask_radius = np.sqrt((X_full['jmuon_pos_x'] - centre_x)**2 + (X_full['jmuon_pos_y'] - centre_y)**2) < approx_r*radius_crit
threshold_likelihood = np.percentile(X_full['jmuon_likelihood'], likelihood_crit)
mask_likelihood = X_full['jmuon_likelihood'] > threshold_likelihood
mask = mask_z_dir & mask_pos_dir_z & mask_likelihood & mask_energy & mask_radius

X_full = X_full[mask]
y_full = y_full[mask]

neutrino_count_post_selection = y_full["label"].sum()

print("Neutrino count post selection ",neutrino_count_post_selection)
print("ratio ",neutrino_count_post_selection/initial_neutr_count)

kfold = KFold(n_splits=5, shuffle=True)
exp_data = pd.DataFrame()

rev_ordered_features = ['jmuon_JSTART_NPE_MIP', 'jmuon_JGANDALF_BETA0_RAD', 'jmuon_AASHOWERFIT_ENERGY', 'jmuon_JGANDALF_BETA1_RAD', 'jmuon_AASHOWERFIT_NUMBER_OF_HITS', 'jmuon_JGANDALF_LAMBDA', 'jmuon_JENERGY_NUMBER_OF_HITS', 'jmuon_t', 'jmuon_dir_x', 'jmuon_dir_y', 'jmuon_JENERGY_MAXIMAL_ENERGY', 'jmuon_JGANDALF_NUMBER_OF_ITERATIONS', 'jmuon_JENERGY_NDF', 'jmuon_JENERGY_NOISE_LIKELIHOOD', 'jmuon_JSHOWERFIT_ENERGY', 'jmuon_JENERGY_ENERGY', 'jmuon_JENERGY_MUON_RANGE_METRES', 'jmuon_E', 'jmuon_JGANDALF_NUMBER_OF_HITS', 'jmuon_JENERGY_CHI2', 'jmuon_JGANDALF_CHI2', 'jmuon_dir_z', 'jmuon_likelihood', 'jmuon_pos_z', 'jmuon_pos_y', 'jmuon_JENERGY_MINIMAL_ENERGY', 'jmuon_JSTART_NPE_MIP_TOTAL', 'jmuon_pos_x', 'jmuon_JSTART_LENGTH_METRES']

n_features = 14
X_selected_features = X_full[rev_ordered_features[-n_features:]]

max_samples =  best_params["max_samples"]
max_features = best_params["max_features"]
n_estimators = best_params["n_estimators"]
criterion = best_params["criterion"]
min_samples_split = best_params["min_samples_split"]
class_weight = best_params["class_weight"]


for train_index, test_index in kfold.split(X_selected_features):

    X_train, X_test = X_selected_features.iloc[train_index].copy(), X_selected_features.iloc[test_index].copy()
    y_train_compl, y_test_compl = y_full.iloc[train_index].copy(), y_full.iloc[test_index].copy()

    y_train = y_train_compl['label']

    if min_samples_split > len(train_index)*max_samples:
        total_neutrino_efficiency = 0

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
    total_neutrino_efficiency = 0

elif len(muons) == 0:
    total_neutrino_efficiency = neutrino_count_post_selection/initial_neutr_count *100

else:

    muon_contamination_percs = muon_contamination(muons, neutrs)
    neutrino_efficiency_percs = neutrino_efficiency(neutrs)

    try:
        where = np.argwhere(muon_contamination_percs <= 1)
        arg = where[-1]
    except:
        arg = 0

    neutrino_efficiency_at_1 = neutrino_efficiency_percs[arg]
    print("neutrino effiency at 1", neutrino_efficiency_at_1)

    # Calculate the total neutrino efficiency
    total_neutrino_efficiency = neutrino_count_post_selection/initial_neutr_count * neutrino_efficiency_at_1

print(total_neutrino_efficiency)

