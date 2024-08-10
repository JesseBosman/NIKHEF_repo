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

def plot_feature_importances(feature_importances, train_columns, det, save_dir):

    n_features = 15
    indices = np.argsort(feature_importances)[::-1]
    print("Feature ranking:")
    for f in range(len(train_columns)):
        print("%d. feature %s (%f)" % (f + 1, train_columns[indices[f]], feature_importances[indices[f]]))
    # Get feature importances from the RandomForestClassifier


    # Get column names from X_train

    # Rearrange column names based on sorted feature importances
    sorted_column_names = train_columns[indices[:n_features]]
    sorted_importances = feature_importances[indices[:n_features]]

    # Create a bar plot

    plt.barh(range(n_features), sorted_importances)
    plt.yticks(range(n_features), sorted_column_names)
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importances Top {} Features'.format(n_features))
    plt.savefig(save_dir+"feature_importances.png", bbox_inches = 'tight')
    plt.close()

def contamination_vs_desired_efficiency(muons, neutrs, new_scores = False,  start= 0.001, stop= 1., n= 1000):
    
    score_thresholds = np.linspace(start,stop,n)
    contamination_percs = []
    neutrino_efficiency_percs = []
    total_neutrs = np.sum(neutrs["weight_one_year"])
    if new_scores:
        label = "muonscore_new"
    else:
        label = "muonscore"

    for threshold in score_thresholds:
            
        rem_neutrs = np.sum(neutrs[(neutrs[label]<=threshold)]["weight_one_year"])
        neutrino_efficiency = rem_neutrs/total_neutrs*100
        rem_muons = np.sum(muons[muons[label]<=threshold]["weight_one_year"])
        contamination_perc = rem_muons/(rem_muons+rem_neutrs)*100
        contamination_percs.append(contamination_perc)
        neutrino_efficiency_percs.append(neutrino_efficiency)
    

    return np.array(contamination_percs), np.array(neutrino_efficiency_percs), score_thresholds

def plot_efficiency_contamination(old_contamination, new_contamination, old_efficiency, new_efficiency, save_dir):

    fig, ax1 = plt.subplots()

    # Plot neutrino efficiency
    ax1.plot(old_contamination, old_efficiency, label='old', color= "red")
    ax1.plot(new_contamination, new_efficiency, label='new', color= "blue")
    ax1.set_xlabel('Muon Contamination [%]')
    ax1.axvline(x=2.9e-1, color='black', linestyle='--', label = "threshold")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(40, 100)
    ax1.set_ylabel('Neutrino Efficiency [%]')
    plt.legend(loc = "best") 

    plt.title('Contamination vs Efficiency')
    plt.grid(True)
    plt.savefig(save_dir+"contamination_vs_efficiency.png", bbox_inches = 'tight')
    plt.show()





def score_histogram(exp_data, det, save_dir, new_score = False):
    muons = exp_data[exp_data['is_neutrino'] == 0]
    neutrinos = exp_data[exp_data['is_neutrino'] == 1]

    if new_score:
        hist_muon_scores_muons, score_bins = np.histogram(muons['muonscore_new'], bins = 100, range = (0,1))
        hist_muon_scores_neutrs = np.histogram(neutrinos['muonscore_new'], bins = 100, range = (0,1))[0]
        save_path = save_dir+ "score_histogram_new.png"
    else:
        hist_muon_scores_muons, score_bins = np.histogram(muons['muonscore'], bins = 100, range = (0,1))
        hist_muon_scores_neutrs = np.histogram(neutrinos['muonscore'], bins = 100, range = (0,1))[0]
        save_path = save_dir+ "score_histogram_old.png"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    frac_muon_score_muons = hist_muon_scores_muons/np.sum(hist_muon_scores_muons)
    frac_muon_score_neutrs = hist_muon_scores_neutrs/np.sum(hist_muon_scores_neutrs)


    plt.figure()
    plt.yscale('log')
    plt.bar(score_bins[:-1], frac_muon_score_muons, width = 1/100, alpha = 0.2, label = "Muons")
    plt.bar(score_bins[:-1], frac_muon_score_neutrs, width = 1/100, alpha = 0.2, label = "Neutrinos")
    plt.ylabel("Fraction of events")
    plt.xlabel("Muon score")
    plt.ylim(10**-5, 1)
    plt.xlim(-0.01,1)
    plt.grid()
    plt.legend()
    plt.savefig(save_path, bbox_inches = 'tight')
    plt.close()

def figure_of_merit(exp_data, start, stop, n, new_score = False):
    muon_contamination_target = 0.0029 # per analysis of what the threshold achieves based on old muonscore

    for threshold in np.linspace(start, stop, n):
        muon_contamination , neutrino_efficiency = get_cont_and_eff_at_threshold(exp_data, threshold, new_score)
        if muon_contamination >= muon_contamination_target:
            break
    
    print("Muon score threshold was: ", threshold)
    return neutrino_efficiency


def tables_of_merit(exp_data, det, new_score = False):
    energy_reco_bins = [[2.0, 4.0, 4.61340151, 5.32086836, 6.13682553, 7.07791004, 8.1633102, 9.4151569, 10.85897475, 12.52420262, 14.4447938, 16.65990837, 19.21471159, 22.16129485, 25.55973776, 29.47933316, 34.0, 50, 100, 1000]]
    cost_reco_bins =  [-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0]
    if not new_score:
        data_HP_tracks = exp_data[exp_data['sel_HP_track'] == 1]
        data_LP_tracks = exp_data[exp_data['sel_LP_track'] == 1]
        data_showers = exp_data[exp_data['sel_shower'] == 1]

    else:
        data_HP_tracks = exp_data[(exp_data["muonscore_new"]<1.5e-4)&(exp_data['trackscore'] >0.56)&(exp_data['energy_recoJEnergy']>=2)&(exp_data['energy_recoJEnergy']<50)]
        data_LP_tracks = exp_data[(exp_data["muonscore_new"]>=1.5e-4)&(exp_data['trackscore'] >0.56)&(exp_data['energy_recoJEnergy']>=2)&(exp_data['energy_recoJEnergy']<50)]
        data_showers = exp_data[(exp_data['trackscore'] <=0.56)&(exp_data['energy_recoJShower']>=2)&(exp_data['energy_recoJShower']<1000)&(exp_data["cos_zenith_recoJShower"]<0)]


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

    # max_features = trial.suggest_categorical("max_features", [None, "sqrt", "log2"])
    # n_estimators = trial.suggest_int("n_estimators", 101, 501, step = 50)
    # max_samples = trial.suggest_float("max_samples", 0.2, 0.95)
    # min_samples_split = trial.suggest_int("min_samples_split", 2, 15)
    # min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
    # max_depth = trial.suggest_int("max_depth", 5, 50, step = 5)
    # class_weight = trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample", None])
    # criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
    # n_features = trial.suggest_int("n_features", 5, 50, step = 5)

    random_state = 42


    max_features = "sqrt"
    n_estimators = 101
    max_samples = 0.6
    min_samples_split = 2
    min_samples_leaf = 1
    max_depth = 10
    class_weight = "balanced"
    criterion = "gini"
    n_features = 30

    det = "ORCA10"


    save_dir = "/data/antares/users/jwbosman/results/{}/RDF/post_cuts_mf{}_nest{}_ms{}_mss{}_msl{}_md{}_cw{}_cr{}_nf{}/".format(det, max_features, n_estimators, max_samples, min_samples_split, min_samples_leaf, max_depth, class_weight, criterion, n_features)
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

        y_train = y_train_compl['is_neutrino']

        clf = RFC(random_state=random_state, max_features=max_features,
                   n_estimators=n_estimators, max_samples=max_samples, min_samples_split=min_samples_split,
                     min_samples_leaf=min_samples_leaf, max_depth=max_depth, class_weight=class_weight, criterion=criterion)   

        # clf = RFC(random_state=random_state, n_estimators = 101) # baseline model

            

        clf.fit(X_train, y_train)
        importances += clf.feature_importances_
        prediction_probabilities = clf.predict_proba(X_test)
        y_pred = np.argmax(prediction_probabilities, axis = 1)

        y_test_compl['muonscore_new'] = prediction_probabilities[:,0]
        y_test_compl['prediction'] = y_pred
        # y_test_compl["energy_recoJEnergy"] = X_test["energy_recoJEnergy"]
        # y_test_compl["energy_recoJShower"] = X_test["energy_recoJShower"]
        # y_test_compl[""] = X_test['jmuon_E']

        exp_data = pd.concat([exp_data, y_test_compl])

    importances = importances/5
    X_train, X_test, y_train_compl, y_test_compl  = None, None, None, None

    score_histogram(exp_data, det, save_dir, True)
    plot_feature_importances(importances, X_full.columns, det, save_dir)
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



