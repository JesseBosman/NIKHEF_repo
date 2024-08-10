import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import optuna
import os
from xgboost import XGBClassifier
from RDF import *
from functools import partial


def objective(trial, data, simulation_columns, top_50_relevant_columns_basemodel, random_state):

    booster = trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"])
    eta = trial.suggest_float("eta", 0.1, 0.6)
    gamma = trial.suggest_float("gamma", 0, 0.5)
    max_depth = trial.suggest_int("max_depth", 3, 10)
    min_child_weight = trial.suggest_int("min_child_weight", 1, 10)
    max_delta_step = 0
    subsample = trial.suggest_float("subsample", 0.4, 1)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.3, 1)
    colsample_bylevel = trial.suggest_float("colsample_bylevel", 0.3, 1)
    colsample_bynode = trial.suggest_float("colsample_bynode", 0.3, 1)
    reg_lambda = trial.suggest_float("reg_lambda", 0, 2)
    alpha = trial.suggest_float("alpha", 0, 2)
    tree_method = trial.suggest_categorical("tree_method", ["hist", "exact", "approx"])
    scale_pos_weight = trial.suggest_float("scale_pos_weight", 0.2, 5)
    n_features = trial.suggest_int("n_features", 5, 50, step = 5)


    # save_dir = "/data/antares/users/jwbosman/results/{}/XGDC/post_cuts_bo{}_et{}_ga{}_md{}_mcw{}_mds{}_ss{}_cbt{}_cbl{}_cbn{}_rl{}_al{}_tm{}_spw{}_npt{}_nf{}/".format(det, booster, eta, gamma, max_depth, min_child_weight, max_delta_step, subsample, colsample_bytree, colsample_bylevel, colsample_bynode, reg_lambda, alpha, tree_method, scale_pos_weight, num_parallel_tree, n_features)
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)

    y_full = data[simulation_columns]
    data.drop(columns = simulation_columns, inplace = True)
    X_full = data.copy()
    X_full = X_full[top_50_relevant_columns_basemodel[:n_features]]
    data = None

  

    SKF = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    exp_data = pd.DataFrame()


    for train_index, test_index in SKF.split(X_full, y_full['is_neutrino']):

        X_train, X_test = X_full.iloc[train_index].copy(), X_full.iloc[test_index].copy()
        y_train_compl, y_test_compl = y_full.iloc[train_index].copy(), y_full.iloc[test_index].copy()

        y_train = y_train_compl['is_neutrino'].to_numpy()

        clf = XGBClassifier(booster = booster, verbosity = 1, eta = eta, gamma = gamma, max_depth = max_depth, min_child_weight = min_child_weight, max_delta_step = max_delta_step, subsample = subsample, colsample_bytree = colsample_bytree, colsample_bylevel = colsample_bylevel, colsample_bynode = colsample_bynode, reg_lambda = reg_lambda, alpha = alpha, tree_method = tree_method, scale_pos_weight = scale_pos_weight, seed = random_state)

            
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

    # score_histogram(exp_data, det, save_dir, True)

    start = 0.0
    stop = 0.03
    n = 200
    
    # muons = exp_data[exp_data['is_neutrino'] == 0]
    # neutrinos = exp_data[exp_data['is_neutrino'] == 1]
    # old_contamination, old_efficiency, score_thresholds = contamination_vs_desired_efficiency(muons, neutrinos, new_scores = False, start = start, stop = stop, n= n )
    # new_contamination, new_efficiency, score_thresholds = contamination_vs_desired_efficiency(muons, neutrinos, new_scores = True , start = start, stop = stop, n= n)

    # plot_efficiency_contamination(old_contamination, new_contamination, old_efficiency, new_efficiency, save_dir)

    neutrino_efficiency = figure_of_merit(exp_data, new_score = True, start= start, stop= stop, n= n)
    print("Neutrino efficiency: ", neutrino_efficiency)
    return neutrino_efficiency


if __name__ == "__main__":
    plt.rcParams["figure.figsize"] = (7,7)
    plt.rcParams["figure.labelsize"] = 14
    plt.rcParams["figure.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 12
    det = "ORCA10"
    num_parallel_tree =  1

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
    
    random_state = 42
    objective = partial(objective, data = data, simulation_columns = simulation_columns, top_50_relevant_columns_basemodel = top_50_relevant_columns_basemodel, random_state = random_state)
    
    study_name = "XGDC_study_{}".format(det)  
    storage_name = "sqlite:///{}.db".format(study_name)

    study = optuna.create_study(study_name = study_name, direction="maximize", storage= storage_name, load_if_exists= True)
    study.optimize(objective, n_trials=5)

    print(study.best_params)
    print("best total neutrino efficiency at ~0.3% muon contamination = ", study.best_value)



    

     

