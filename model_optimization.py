import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import optuna
from xgboost import XGBClassifier
from functools import partial
import joblib
from functions import list_of_ordered_features_basemodel, figure_of_merit
import argparse
from sklearn.ensemble import HistGradientBoostingClassifier as HGBC

def objective(trial, data, simulation_columns, ordered_features_basemodel, random_state, event_type, classifier = "XGBC"):
    if classifier == "XGBC":
        eta = trial.suggest_float("eta", 0.1, 1.0)
        gamma = trial.suggest_float("gamma", 0, 0.5)
        max_depth = trial.suggest_int("max_depth", 3, 15)
        min_child_weight = trial.suggest_int("min_child_weight", 1, 10)
        subsample = trial.suggest_float("subsample", 0.3, 1)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.2, 1)
        colsample_bylevel = trial.suggest_float("colsample_bylevel", 0.2, 1)
        colsample_bynode = trial.suggest_float("colsample_bynode", 0.2, 1)
        reg_lambda = trial.suggest_float("reg_lambda", 0, 2.5)
        alpha = trial.suggest_float("alpha", 0, 2.5)
        scale_pos_weight = trial.suggest_float("scale_pos_weight", 0.2, 5)
        n_features = trial.suggest_int("n_features", 5, 110, step = 1)
        grow_policy = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
    
    elif classifier == "HGBC":
        learning_rate = trial.suggest_float("learning_rate", 0.1, 1.0)
        max_iter = 101
        max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 11, 101)
        max_depth = trial.suggest_int("max_depth", 3, 15)
        l2_regularization = trial.suggest_float("l2_regularization", 0, 2.5)
        max_features = trial.suggest_float("max_features", 0.2, 1)
        class_weight = trial.suggest_categorical("class_weight", ["balanced", None])
        scoring = trial.suggest_categorical("scoring", ["roc_auc", "balanced_accuracy", "f1", "log_loss"])

    y_full = data[simulation_columns]
    X_full = data[ordered_features_basemodel[:n_features]]
    SKF = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    exp_data = pd.DataFrame()


    for train_index, test_index in SKF.split(X_full, y_full['is_neutrino']):

        X_train, X_test = X_full.iloc[train_index].copy(), X_full.iloc[test_index].copy()
        y_train_compl, y_test_compl = y_full.iloc[train_index].copy(), y_full.iloc[test_index].copy()

        y_train = y_train_compl['is_neutrino']

        if classifier == "XGBC":
            clf = XGBClassifier(grow_policy = grow_policy, verbosity = 1, eta = eta, gamma = gamma, max_depth = max_depth, min_child_weight = min_child_weight, subsample = subsample, colsample_bytree = colsample_bytree, colsample_bylevel = colsample_bylevel, colsample_bynode= colsample_bynode, reg_lambda = reg_lambda, alpha = alpha, scale_pos_weight = scale_pos_weight, seed = random_state)

        elif classifier == "HGBC":
            clf = HGBC(learning_rate = learning_rate, max_iter = max_iter, max_leaf_nodes = max_leaf_nodes, max_depth = max_depth, l2_regularization = l2_regularization, max_features = max_features, random_state = random_state,scoring = scoring, verbose = 1, class_weight = class_weight)
            
        clf.fit(X_train, y_train)
        prediction_probabilities = clf.predict_proba(X_test)
        y_pred = np.argmax(prediction_probabilities, axis = 1)

        y_test_compl['muonscore_new'] = prediction_probabilities[:,0]
        y_test_compl['prediction'] = y_pred

        exp_data = pd.concat([exp_data, y_test_compl])

    X_train, X_test, y_train_compl, y_test_compl  = None, None, None, None


    start = -7
    stop = -0.7
    n = 600
    

    if event_type == "track":
        muon_contamination_target = 2.71
        neutrino_efficiency_LP, score_threshold = figure_of_merit(exp_data, new_score = True, start= start, stop= stop, n= n, muon_contamination_target= muon_contamination_target)
        muon_contamination_target = 0
        neutrino_efficiency_HP, score_threshold = figure_of_merit(exp_data, new_score = True, start= start, stop= stop, n= n, muon_contamination_target = muon_contamination_target)
        score = neutrino_efficiency_HP + 0.1*neutrino_efficiency_LP
    
    elif event_type == "shower":
        muon_contamination_target = 2.24
        score, score_threshold = figure_of_merit(exp_data, new_score = True, start= start, stop= stop, n= n, muon_contamination_target= muon_contamination_target)

    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--det", type = str, default = "ORCA10")
    parser.add_argument("--event_type", type = str, default = "track", choices = ["track", "shower"])
    parser.add_argument("--n_trials", type = int, default = 2)
    parser.add_argument("--classifier", type = str, default = "XGBC", choices =["XGBC", "HGBC"])
    args = parser.parse_args()
    det = args.det
    event_type = args.event_type
    n_trials = args.n_trials
    classifier = args.classifier

    ordered_features_basemodel = list_of_ordered_features_basemodel(event_type)
    data = pd.read_hdf("/data/antares/users/jwbosman/{}/train_{}_data.h5".format(det , event_type))
    simulation_columns = ['mc_id', 'run_id', 'T.sum_mc_evt.E_max_gen',
                            'T.sum_mc_evt.weight', 'T.sum_mc_evt.weight_noOsc', 'weight', 'exposure', 
                            'weight_rate', 'type', 'run_duration', 'is_cc', 'is_neutrino', 'cos_zenith_true', 
                           'bjorken_y_true', 'w1', 'w2', 'ngen', 'E_min_gen', 'weight_one_year', 'pdgid', 'muonscore', 
                           'runs_neutrino2024_veto_sparks', 'E.frame_index', "E.trigger_counter", "EG", 
                            "isJsirene", "sel_HP_track", "sel_LP_track", "sel_shower",
                            "run_duration", "pos_x", "pos_y", "pos_z", "energy", "int_len"]
    
    random_state = 42
    objective = partial(objective, data = data, simulation_columns = simulation_columns, ordered_features_basemodel = ordered_features_basemodel, random_state = random_state, event_type = event_type, classifier = classifier)
    
    study_name = "{}_study_{}".format(classifier, event_type)  
    try:
        study = joblib.load("/data/antares/users/jwbosman/NIKHEF_repo/{}.pkl".format(study_name))
    except:
        study = optuna.create_study(study_name = study_name , direction = "maximize")
   
    
    print("Number of finished trials: ", len(study.trials))
    study.optimize(objective, n_trials=n_trials)

    joblib.dump(study, "/data/antares/users/jwbosman/NIKHEF_repo/{}.pkl".format(study_name))




    

     

