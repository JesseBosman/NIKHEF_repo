import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import StratifiedKFold
from scipy.stats import randint
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import optuna
import os
import optuna
from data_analysis import get_cont_and_eff_at_threshold
from xgboost import XGBClassifier
from RDF import *


if __name__ == "__main__":
    plt.rcParams["figure.figsize"] = (7,7)
    plt.rcParams["figure.labelsize"] = 14
    plt.rcParams["figure.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 12

    random_state = 42
    # booster = "gbtree"
    # verbosity = 1
    # eta = 0.3
    # gamma = 0
    # max_depth = 6
    # min_child_weight = 1
    # max_delta_step = 0
    # subsample = 1
    # colsample_bytree = 1
    # colsample_bylevel = 1
    # colsample_bynode = 1
    # reg_lambda = 1
    # alpha = 0
    # tree_method = "auto"
    # scale_pos_weight = 1
    # num_parallel_tree =  1
    # n_features = 50

    det = "ORCA10"

    study_name = "XGDC_study_{}".format(det)  
    storage_name = "sqlite:///{}.db".format(study_name)

    study = optuna.load_study(study_name=study_name, storage=storage_name)

    best_params = study.best_params

    print("Best parameters: \n")
    print(best_params)
    print("With best value: \n")
    print(study.best_value)

    booster = best_params["booster"]
    verbosity = 1
    eta = best_params["eta"]
    gamma = best_params["gamma"]
    max_depth = best_params["max_depth"]
    min_child_weight = best_params["min_child_weight"]
    max_delta_step = best_params["max_delta_step"]
    subsample = best_params["subsample"]
    colsample_bytree = best_params["colsample_bytree"]
    colsample_bylevel = best_params["colsample_bylevel"]
    colsample_bynode = best_params["colsample_bynode"]
    reg_lambda = best_params["reg_lambda"]
    alpha = best_params["alpha"]
    tree_method = best_params["tree_method"]
    scale_pos_weight = best_params["scale_pos_weight"]
    num_parallel_tree =  1
    n_features = best_params["n_features"]

    


    save_dir = "/data/antares/users/jwbosman/results/{}/XGDC/post_cuts_bo{}_et{}_ga{}_md{}_mcw{}_mds{}_ss{}_cbt{}_cbl{}_cbn{}_rl{}_al{}_tm{}_spw{}_npt{}/".format(det, booster, eta, gamma, max_depth, min_child_weight, max_delta_step, subsample, colsample_bytree, colsample_bylevel, colsample_bynode, reg_lambda, alpha, tree_method, scale_pos_weight, num_parallel_tree)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    top_50_relevant_columns_basemodel = ['E.trks.dir.z[:,1]', 'cos_zenith_recoJShower',
       'T.feat_Neutrino2020.cherCond_n_hits_dnMup',
       'T.feat_Neutrino2020.cherCond_n_hits_trig_dnMup',
       'T.feat_Neutrino2020.cherCond_hits_meanZposition',
       'T.feat_Neutrino2020.cherCond_hits_trig_meanZposition',
       'angle_shfit_gandalf', 'T.feat_Neutrino2020.cherCond_n_hits_trig_dnf',
       'meanZhitTrig', 'T.feat_Neutrino2020.cherCond_n_hits_dnf',
       'T.feat_Neutrino2020.cherCond_n_hits_upf', 'pos_r_JGandalf',
       'E.trks.fitinf[:,0,2]', 'likelihood_JGandalf',
       'T.feat_Neutrino2020.QupOvernHits',
       'T.feat_Neutrino2020.zClosestApproach', 'pos_z_recoJShower',
       'T.sum_jppshower.prefit_posfit_distance', 'pos_z_recoJGandalf',
       'ratio_E_jshf_gandalf', 'maximumToT_triggerHit',
       'gandalf_shfit_lik_ratio', 'loglik_jg',
       'T.feat_Neutrino2020.gandalf_Qup',
       'T.feat_Neutrino2020.n_hits_earlyTrig',
       'T.feat_Neutrino2020.cherCond_n_hits_trig_upf',
       'T.feat_Neutrino2020.QupMinusQdn', 'closest[:,1,1]',
       'T.sum_hits.nlines', 'T.sum_jppshower.prefit_posfit_dt',
       'closest[:,0,0]', 'sumtot[:,1,1]',
       'T.feat_Neutrino2020.cherCond_n_doms',
       'T.feat_Neutrino2020.dClosestApproach', 'min_dom_dist',
       'energy_recoJEnergy', 'E.trks.dir.y[:,1]', 'E.trks.fitinf[:,0,13]',
       'crkv_nhits50[:,1,1]', 'T.sum_trig_hits.atot', 'T.sum_hits.nhits',
       'trackscore', 'closest[:,1,0]', 'crkv_nhits[:,1,0]', 'furthest[:,1,1]',
       'E.trks.dir.z[:,0]', 'T.sum_hits.ndoms', 'crkv_nhits[:,1,1]',
       'cos_zenith_recoJGandalf', 'energy_recoJShower']
    
    data = pd.read_hdf("/data/antares/users/jwbosman/{}/post_general_selection_cuts_data.h5".format(det))   
    simulation_columns = ['mc_id', 'E.mc_trks.dir.z[:,0]', 'run_id', 'rectype_JShower', 'T.sum_mc_evt.E_max_gen',
                           'T.sum_mc_evt.E_min_gen', 'T.sum_mc_evt.livetime_DAQ', 'T.sum_mc_evt.weight', 'T.sum_mc_evt.weight_noOsc', 
                           'weight', 'exposure', 'weight_rate', 'type', 'run_duration', 'is_cc', 'is_neutrino', 'cos_zenith_true', 
                           'bjorken_y_true', 'w1', 'w2', 'ngen', 'E_min_gen', 'weight_one_year', 'pdgid', 'muonscore', 'runs_neutrino2024_veto_sparks', 'E.frame_index', "E.trigger_counter", "EG", "E.mc_trks.dir.z[:,0]", 
                            "isJsirene", "sel_HP_track", "sel_LP_track", "sel_shower",
                            "run_duration", "pos_x", "pos_y", "pos_z", "energy", "int_len"]
    y_full = data[simulation_columns]
    data.drop(columns = simulation_columns, inplace = True)
    X_full = data.copy()
    X_full = X_full[top_50_relevant_columns_basemodel[:n_features]]

    data = None    

    SKF = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    exp_data = pd.DataFrame()
    importances = np.zeros(len(X_full.columns))

    for train_index, test_index in SKF.split(X_full, y_full['is_neutrino']):

        X_train, X_test = X_full.iloc[train_index].copy(), X_full.iloc[test_index].copy()
        y_train_compl, y_test_compl = y_full.iloc[train_index].copy(), y_full.iloc[test_index].copy()

        y_train = y_train_compl['is_neutrino'].to_numpy()

        clf = XGBClassifier(booster = booster, verbosity = verbosity, eta = eta, gamma = gamma, max_depth = max_depth, min_child_weight = min_child_weight, max_delta_step = max_delta_step, subsample = subsample, colsample_bytree = colsample_bytree, colsample_bylevel = colsample_bylevel, colsample_bynode = colsample_bynode, reg_lambda = reg_lambda, alpha = alpha, tree_method = tree_method, scale_pos_weight = scale_pos_weight, seed = random_state)

            
        clf.fit(X_train.to_numpy(), y_train)
        prediction_probabilities = clf.predict_proba(X_test.to_numpy())
        y_pred = np.argmax(prediction_probabilities, axis = 1)

        y_test_compl['muonscore_new'] = prediction_probabilities[:,0]
        y_test_compl['prediction'] = y_pred
        # y_test_compl["energy_recoJEnergy"] = X_test["energy_recoJEnergy"]
        # y_test_compl["energy_recoJShower"] = X_test["energy_recoJShower"]
        # y_test_compl[""] = X_test['jmuon_E']

        exp_data = pd.concat([exp_data, y_test_compl])

    X_train, X_test, y_train_compl, y_test_compl  = None, None, None, None

    score_histogram(exp_data, det, save_dir, True)
    

    start = 0.0
    stop = 0.03
    n = 200
    
    muons = exp_data[exp_data['is_neutrino'] == 0]
    neutrinos = exp_data[exp_data['is_neutrino'] == 1]
    old_contamination, old_efficiency, score_thresholds = contamination_vs_desired_efficiency(muons, neutrinos, new_scores = False, start = start, stop = stop, n= n )
    new_contamination, new_efficiency, score_thresholds = contamination_vs_desired_efficiency(muons, neutrinos, new_scores = True , start = start, stop = stop, n= n)

    plot_efficiency_contamination(old_contamination, new_contamination, old_efficiency, new_efficiency, save_dir)

    neutrino_efficiency = figure_of_merit(exp_data, new_score = True, start= start, stop= stop, n= n)
    print("Neutrino efficiency: ", neutrino_efficiency)


