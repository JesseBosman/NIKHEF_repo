import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import HistGradientBoostingClassifier as HGBC
from sklearn.model_selection import StratifiedKFold
from scipy.stats import randint
import matplotlib.pyplot as plt
import os
from functions import list_of_ordered_features_basemodel, score_histogram, plot_feature_importances, contamination_vs_desired_efficiency, plot_efficiency_contamination, table_of_merit, figure_of_merit, histogram_2d
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
import time


    

if __name__ == "__main__":
    plt.rcParams["figure.figsize"] = (7,7)
    plt.rcParams["figure.labelsize"] = 14
    plt.rcParams["figure.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 12

    simulation_columns = ['mc_id', 'run_id', 'T.sum_mc_evt.E_max_gen',
                            'T.sum_mc_evt.weight', 'T.sum_mc_evt.weight_noOsc', 'weight', 'exposure', 
                            'weight_rate', 'type', 'run_duration', 'is_cc', 'is_neutrino', 'cos_zenith_true', 
                           'bjorken_y_true', 'w1', 'w2', 'ngen', 'E_min_gen', 'weight_one_year', 'pdgid', 'muonscore', 
                           'runs_neutrino2024_veto_sparks', 'E.frame_index', "E.trigger_counter", "EG", 
                            "isJsirene", "sel_HP_track", "sel_LP_track", "sel_shower",
                            "run_duration", "pos_x", "pos_y", "pos_z", "energy", "int_len"]

    random_state = 42

    det = "ORCA10"
    event_type = "track"
    classifier = "XGBC"
    baseline_or_optuna = "optuna"
    
   
    test_data = pd.read_hdf("/data/antares/users/jwbosman/{}/test_{}_data.h5".format(det , event_type))
    y_test = test_data[simulation_columns]
    test_data.drop(columns = simulation_columns, inplace = True)
    X_test = test_data.copy()
    test_data = None

    start_time_test = time.time()

    train_data = pd.read_hdf("/data/antares/users/jwbosman/{}/train_{}_data.h5".format(det , event_type))
    y_train = train_data[simulation_columns]
    train_data.drop(columns = simulation_columns, inplace = True)
    X_train = train_data.copy()
    train_data = None

    
    if baseline_or_optuna == "optuna":

        study_name = "{}_study_{}".format(classifier, event_type)  
        try:
            study = joblib.load("/data/antares/users/jwbosman/NIKHEF_repo/{}.pkl".format(study_name))
            params = study.best_params
            
            best_trial = study.best_trial.number
            n_trials = len(study.trials)
            print("Best trial found after {} trials is: {}".format(n_trials, best_trial))
            print("Best parameters found are: ", params)
            raise ValueError("No study found with name: ", study_name)
            ordered_features_basemodel = list_of_ordered_features_basemodel(event_type=event_type)
            save_dir_test = "/data/antares/users/jwbosman/results/{}/{}/{}/best_trial_{}_after_{}_trials/test/".format(det, classifier, event_type, best_trial, n_trials)
            save_dir_train = "/data/antares/users/jwbosman/results/{}/{}/{}/best_trial_{}_after_{}_trials/train/".format(det, classifier, event_type, best_trial, n_trials)

            X_train = X_train[ordered_features_basemodel[:params["n_features"]]]
            X_test = X_test[ordered_features_basemodel[:params["n_features"]]]
            if classifier == "RDF":
                clf = RFC(random_state=random_state, n_jobs = -1, **params)
            elif classifier == "XGBC":
                
                clf = XGBClassifier(seed = random_state, **params)
            elif classifier == "GBC":
                clf = GBC(random_state=random_state, **params)
            elif classifier == "HGBC":
                clf = HGBC(random_state=random_state, **params)
            else:
                raise ValueError("Classifier not recognized")
        except:
            raise ValueError("No study found with name: ", study_name)
    
    elif baseline_or_optuna == "baseline":
        save_dir_test = "/data/antares/users/jwbosman/results/{}/{}/{}/basemodel/test/".format(det, classifier, event_type)
        save_dir_train = "/data/antares/users/jwbosman/results/{}/{}/{}/basemodel/train/".format(det, classifier, event_type)
        if classifier == "RDF":
            clf = RFC(random_state=random_state, n_estimators = 101, n_jobs = -1)

        elif classifier == "XGBC":
            
            clf = XGBClassifier(seed = random_state)

        elif classifier == "GBC":
            clf = GBC(random_state=random_state, n_estimators = 101)
        elif classifier == "HGBC":
            clf = HGBC(random_state=random_state, max_iter = 101)

        else:
            raise ValueError("Classifier not recognized")
        
    if not os.path.exists(save_dir_test):
        os.makedirs(save_dir_test)
    if not os.path.exists(save_dir_train):
        os.makedirs(save_dir_train)
    
    # RESULTS FOR TEST DATA

    clf.fit(X_train, y_train['is_neutrino'])
    exp_data = y_test.copy()
    prediction_probabilities = clf.predict_proba(X_test)
    y_pred = np.argmax(prediction_probabilities, axis = 1)

    exp_data['muonscore_new'] = prediction_probabilities[:,0]
    exp_data['prediction'] = y_pred

    # score_histogram(exp_data, det, save_dir_test, True)
    print("Results for {} data".format(event_type))
    if (classifier == "RDF") or (classifier == "GBC") or (classifier == "XGBC"):
        importances = clf.feature_importances_
        plot_feature_importances(importances, X_test.columns, det, save_dir_test)

    start = -7
    stop = -0.7
    n = 600
    
    muons = exp_data[exp_data['is_neutrino'] == 0]
    neutrinos = exp_data[exp_data['is_neutrino'] == 1]

    min_1 = exp_data['muonscore_new'].min()
    max_1 = exp_data['muonscore_new'].max()
    min_2 = exp_data['energy'].min()
    max_2 = exp_data['energy'].max()

    histogram_2d(data = neutrinos, column_1= "muonscore_new", column_2="energy", min_1 = min_1, max_1=max_1, min_2= min_2, max_2= max_2, col_1_log= True, col_2_log= True,save_dir= save_dir_test, type = "neutrino")
    histogram_2d(data = muons, column_1= "muonscore_new", column_2="energy", min_1 = min_1, max_1=max_1, min_2= min_2, max_2= max_2, col_1_log= True, col_2_log= True,save_dir= save_dir_test, type = "muon")

    old_contamination, old_efficiency, score_thresholds = contamination_vs_desired_efficiency(muons, neutrinos, new_scores = False, start = start, stop = stop, n= n )
    new_contamination, new_efficiency, score_thresholds = contamination_vs_desired_efficiency(muons, neutrinos, new_scores = True , start = start, stop = stop, n= n)

    plot_efficiency_contamination(old_contamination, new_contamination, old_efficiency, new_efficiency, save_dir_test, event_type, train_or_test= "test")

    if event_type == "track":
        muon_contamination_target = 2.66
        neutrino_efficiency, score_threshold_LP = figure_of_merit(exp_data, new_score = True, start= start, stop= stop, n= n, muon_contamination_target= muon_contamination_target)
        muon_contamination_target = 0
        neutrino_efficiency, score_threshold_HP = figure_of_merit(exp_data, new_score = True, start= start, stop= stop, n= n, muon_contamination_target = muon_contamination_target)
        print("Table of merit for {} {} track criterium on test data is:".format(baseline_or_optuna, classifier))
        table_of_merit(exp_data,score_threshold_LP, score_threshold_HP, event_type, new_score = True, save_dir= save_dir_test)
        
    
    elif event_type == "shower":
        muon_contamination_target = 2.08
        neutrino_efficiency, score_threshold = figure_of_merit(exp_data, new_score = True, start= start, stop= stop, n= n, muon_contamination_target= muon_contamination_target)
        print("Table of merit for {} {} shower criterium on test data is:".format(baseline_or_optuna, classifier))
        table_of_merit(exp_data= exp_data ,score_threshold_LP= score_threshold, event_type = event_type, new_score = True, save_dir=save_dir_test)
        
    print("Time taken for test data is: ", time.time() - start_time_test)
    # RESULTS FOR TRAIN DATA
    start_time_train = time.time()
    y = y_train.copy()
    X = X_train.copy()
    exp_data = pd.DataFrame()

    SKF = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    feature_importances = np.zeros(len(X.columns))

    for train_index, test_index in SKF.split(X, y['is_neutrino']):

        X_train, X_test = X.iloc[train_index].copy(), X.iloc[test_index].copy()
        y_train_compl, y_test_compl = y.iloc[train_index].copy(), y.iloc[test_index].copy()

        y_train = y_train_compl['is_neutrino'].to_numpy()

        if baseline_or_optuna == "optuna":
                
            if classifier == "RDF":
                clf = RFC(random_state=random_state, n_jobs = -1, **params)
            elif classifier == "GBC":	
                clf = GBC(random_state=random_state, **params)
            elif classifier == "HGBC":
                clf = HGBC(random_state=random_state, **params)
            else:
                clf = XGBClassifier(seed = random_state, **params)
        
        elif baseline_or_optuna == "baseline":
           
            if classifier == "RDF":
                clf = RFC(random_state=random_state, n_estimators = 101, n_jobs = -1)
            elif classifier == "GBC":
                clf = GBC(random_state=random_state, n_estimators = 101)
            elif classifier == "HGBC":
                clf = HGBC(random_state=random_state, max_iter = 101)

            else:
                clf = XGBClassifier(seed = random_state)

            
        clf.fit(X_train.to_numpy(), y_train)
        if (classifier == "RDF") or (classifier == "GBC") or (classifier == "XGBC"):
            feature_importances += clf.feature_importances_
        prediction_probabilities = clf.predict_proba(X_test.to_numpy())
        y_pred = np.argmax(prediction_probabilities, axis = 1)

        y_test_compl['muonscore_new'] = prediction_probabilities[:,0]
        y_test_compl['prediction'] = y_pred

        exp_data = pd.concat([exp_data, y_test_compl])


    X_train, X_test, y_train_compl, y_test_compl  = None, None, None, None
    score_histogram(exp_data, det, save_dir_train, True)
    if (classifier == "RDF") or (classifier == "GBC") or (classifier == "XGBC"):
        importances = feature_importances/5
        plot_feature_importances(importances, X.columns, det, save_dir_train)

    
    start = -7
    stop = -0.7
    n = 600
    
    muons = exp_data[exp_data['is_neutrino'] == 0]
    neutrinos = exp_data[exp_data['is_neutrino'] == 1]

    min_1 = exp_data['muonscore_new'].min()
    max_1 = exp_data['muonscore_new'].max()
    min_2 = exp_data['energy'].min()
    max_2 = exp_data['energy'].max()

    histogram_2d(data = neutrinos, column_1= "muonscore_new", column_2="energy", min_1 = min_1, max_1=max_1, min_2= min_2, max_2= max_2, col_1_log= True, col_2_log= True,save_dir= save_dir_train, type = "neutrino")
    histogram_2d(data = muons, column_1= "muonscore_new", column_2="energy", min_1 = min_1, max_1=max_1, min_2= min_2, max_2= max_2, col_1_log= True, col_2_log= True,save_dir= save_dir_train, type = "muon")

    old_contamination, old_efficiency, score_thresholds = contamination_vs_desired_efficiency(muons, neutrinos, new_scores = False, start = start, stop = stop, n= n )
    new_contamination, new_efficiency, score_thresholds = contamination_vs_desired_efficiency(muons, neutrinos, new_scores = True , start = start, stop = stop, n= n)

    plot_efficiency_contamination(old_contamination, new_contamination, old_efficiency, new_efficiency, save_dir_train, event_type, train_or_test= "train")

    if event_type == "track":
        muon_contamination_target = 1.70
        neutrino_efficiency, score_threshold_LP = figure_of_merit(exp_data, new_score = True, start= start, stop= stop, n= n, muon_contamination_target= muon_contamination_target)
        muon_contamination_target = 0
        neutrino_efficiency, score_threshold_HP = figure_of_merit(exp_data, new_score = True, start= start, stop= stop, n= n, muon_contamination_target = muon_contamination_target)
        print("Table of merit for {} {} track criterium on train data is:".format(baseline_or_optuna, classifier))
        table_of_merit(exp_data,score_threshold_LP, score_threshold_HP, event_type, new_score = True, save_dir= save_dir_train)
        
    
    elif event_type == "shower":
        muon_contamination_target = 1.42
        neutrino_efficiency, score_threshold = figure_of_merit(exp_data, new_score = True, start= start, stop= stop, n= n, muon_contamination_target= muon_contamination_target)
        print("Table of merit for {} {} shower criterium on train data is:".format(baseline_or_optuna, classifier))
        table_of_merit(exp_data= exp_data, score_threshold_LP= score_threshold, event_type = event_type, new_score = True, save_dir=save_dir_train)

    print("Time taken for train data is: ", time.time() - start_time_train)
