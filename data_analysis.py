import pandas as pd
import numpy as np
from functions import overview_data_table
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from functions import list_of_ordered_features_basemodel, figure_of_merit, histogram_muon_vs_neutrino, get_cont_and_eff_at_threshold, scatter_4d, histogram_2d, table_of_merit, correlation_figure, zoomed_in_hist, hist_muon_vs_neutrino_1_feature

if __name__ == "__main__":
    plt.rcParams["figure.figsize"] = (7,7)
    plt.rcParams["figure.labelsize"] = 14
    plt.rcParams["figure.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 12

    

    n_features = 15
    det = "ORCA10"
    for event_type in ["track", "shower"]:
        for train_or_test in ["train", "test"]:
            print("working on {} {} data".format(train_or_test, event_type))
            data = pd.read_hdf("/data/antares/users/jwbosman/{}/{}_{}_data.h5".format(det, train_or_test, event_type))
            save_dir = "/data/antares/users/jwbosman/results/{}/{}/{}/".format(det, event_type, train_or_test)

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # hist_muon_vs_neutrino_1_feature(data, "energy", save_dir+"/1d_hists/", use_weights=True)
            # min_1 = data['muonscore'].min()
            # max_1 = data['muonscore'].max()
            # min_2 = data['energy'].min()
            # max_2 = data['energy'].max()

            # histogram_2d(data = data[data.is_neutrino == 0], column_1= "muonscore", column_2="energy", min_1 = min_1, max_1=max_1, min_2= min_2, max_2= max_2, col_1_log= True, col_2_log= True,save_dir= save_dir, type = "muons")
            # histogram_2d(data = data[data.is_neutrino == 1], column_1= "muonscore", column_2="energy", min_1 = min_1, max_1=max_1, min_2= min_2, max_2= max_2, col_1_log= True, col_2_log= True,save_dir= save_dir, type = "neutrinos")
            # zoomed_in_hist(data = data, column = "T.feat_Neutrino2020.cherCond_n_hits_dnMup", save_dir = save_dir, width="60%", height="60%", xlims= (-50,150), ylims = None)
            # zoomed_in_hist(data = data, column = "T.feat_Neutrino2020.cherCond_n_hits_trig_dnMup", save_dir = save_dir, width="60%", height="60%", xlims= (-50,150), ylims = None)
            # zoomed_in_hist(data = data, column = "T.sum_jppshower.prefit_posfit_distance", save_dir = save_dir, width="60%", height="60%", xlims= (0,100), ylims = None)
            

            # ordered_list_of_features = list_of_ordered_features_basemodel(event_type)
            # correlation_figure(data, ordered_list_of_features[:15], save_dir, title= "correlation matrix top features {} data".format(event_type))
            print("overview table for {} {} data".format(train_or_test, event_type))
            overview_data_table(data)
            # if event_type == "track":
            #     threshold_LP = 2.9e-3
            #     threshold_HP = 1.5e-4

            # elif event_type == "shower":
            #     threshold_LP = 2.9e-3  

            # # efficiency, threshold = figure_of_merit(data, new_score = False, start = -7, stop = -0.7, n = 600, muon_contamination_target = contamination_target)
            
            # if event_type == "track":
            #     table_of_merit(data,threshold_LP, threshold_HP, event_type, new_score = False, save_dir=save_dir)
                
            
            # elif event_type == "shower":

            #     table_of_merit(exp_data= data ,score_threshold_LP= threshold_LP, event_type = event_type, new_score = False, save_dir=save_dir)
                    
            # histogram_muon_vs_neutrino(data, save_dir+"/1d_hists/", use_weights=False)
            # histogram_muon_vs_neutrino(data, save_dir+"/1d_hists/", use_weights=True)

            # lp_muonscore_threshold = 2.9e-3
            # hp_muonscore_threshold = 1.5e-4

            # lp_muon_contamination, lp_neutrino_efficiency = get_cont_and_eff_at_threshold(data, lp_muonscore_threshold, new_score = False)
            # hp_muon_contamination, hp_neutrino_efficiency = get_cont_and_eff_at_threshold(data, hp_muonscore_threshold, new_score = False)

            # print("for {} {} data".format(train_or_test, event_type))
            # print("For LP, neutrino efficiency is : {:.2f} %, at a muon contamination of : {:.2f} %".format(lp_neutrino_efficiency,lp_muon_contamination))
            # print("For HP, neutrino efficiency is : {:.2f} %, at a muon contamination of : {:.2f} %".format(hp_neutrino_efficiency,hp_muon_contamination))
    # muons = data[data["is_neutrino"]==]
    # neutrinos = data[data["is_neutrino"]==1]

    # scatter_4d(muons, "pos_x_recoJGandalf", "pos_y_recoJGandalf", "pos_z_recoJGandalf", "energy_recoJEnergy", save_dir+"/4d_scatters/", "muons", 20, 45)
    # scatter_4d(muons, "pos_x_recoJGandalf", "pos_y_recoJGandalf", "pos_z_recoJGandalf", "cos_zenith_recoJGandalf", save_dir+"/4d_scatters/", "muons", 20, 45)
    # scatter_4d(muons, "pos_x_recoJGandalf", "pos_y_recoJGandalf", "pos_z_recoJGandalf", "cos_zenith_recoJShower", save_dir+"/4d_scatters/", "muons", 20, 45)
    # scatter_4d(muons, "pos_x_recoJShower", "pos_y_recoJShower", "pos_z_recoJShower", "energy_recoJShower", save_dir+"/4d_scatters/", "muons", 20, 45)
    # scatter_4d(muons, "pos_x_recoJShower", "pos_y_recoJShower", "pos_z_recoJShower", "cos_zenith_recoJShower", save_dir+"/4d_scatters/", "muons", 20, 45)
   

    # scatter_4d(neutrinos, "pos_x_recoJGandalf", "pos_y_recoJGandalf", "pos_z_recoJGandalf", "energy_recoJEnergy", save_dir+"/4d_scatters/", "neutrinos", 20, 45)
    # scatter_4d(neutrinos, "pos_x_recoJGandalf", "pos_y_recoJGandalf", "pos_z_recoJGandalf", "cos_zenith_recoJGandalf", save_dir+"/4d_scatters/", "neutrinos", 20, 45)
    # scatter_4d(neutrinos, "pos_x_recoJGandalf", "pos_y_recoJGandalf", "pos_z_recoJGandalf", "cos_zenith_recoJShower", save_dir+"/4d_scatters/", "neutrinos", 20, 45)
    # scatter_4d(neutrinos, "pos_x_recoJShower", "pos_y_recoJShower", "pos_z_recoJShower", "energy_recoJShower", save_dir+"/4d_scatters/", "neutrinos", 20, 45)
    # scatter_4d(neutrinos, "pos_x_recoJShower", "pos_y_recoJShower", "pos_z_recoJShower", "cos_zenith_recoJShower", save_dir+"/4d_scatters/", "neutrinos", 20, 45)

    # for i in range(n_features):
    #     col_1 = ordered_list_of_features[i]
    #     if (col_1 == "energy_recoJEnergy") or (col_1 == "energy_recoJShower") or (col_1 == "energy"):
    #         col_1_log = True
    #     else:
    #         col_1_log = False

    #     col_1_min = data[col_1].min()
    #     col_1_max = data[col_1].max()
    #     for j in range (i+1, n_features):
    #         col_2 = ordered_list_of_features[j]
    #         if (col_2 == "energy_recoJEnergy") or (col_2 == "energy_recoJShower") or (col_2 == "energy"):
    #             col_2_log = True
    #         else:
    #             col_2_log = False
            
    #         col_2_min = data[col_2].min()
    #         col_2_max = data[col_2].max()
            
    #         histogram_2d(muons, col_1, col_2, col_1_log, col_2_log, col_1_min, col_2_min, col_1_max, col_2_max, save_dir+"/2d_hists/", "muons")
    #         histogram_2d(neutrinos, col_1, col_2, col_1_log, col_2_log, col_1_min, col_2_min, col_1_max, col_2_max, save_dir+"/2d_hists/", "neutrinos")


    # # histogram_2d(muons, "pos_x_recoJGandalf", "pos_y_recoJGandalf", False, False, data["pos_x_recoJGandalf"].min(), data["pos_y_recoJGandalf"].min(), data["pos_x_recoJGandalf"].max(), data["pos_y_recoJGandalf"].max(), save_dir+"/2d_hists/", "muons")
    # # histogram_2d(neutrinos, "pos_x_recoJGandalf", "pos_y_recoJGandalf", False, False, data["pos_x_recoJGandalf"].min(), data["pos_y_recoJGandalf"].min(), data["pos_x_recoJGandalf"].max(), data["pos_y_recoJGandalf"].max(), save_dir+"/2d_hists/", "neutrinos")
    # # histogram_2d(muons, "pos_z_recoJShower","E.trks.dir.z[:,1]", False, False, data["pos_z_recoJShower"].min(), data["E.trks.dir.z[:,1]"].min(), data["pos_z_recoJShower"].max(), data["E.trks.dir.z[:,1]"].max(), save_dir+"/2d_hists/", "muons")
    # # histogram_2d(neutrinos, "pos_z_recoJShower","E.trks.dir.z[:,1]", False, False, data["pos_z_recoJShower"].min(), data["E.trks.dir.z[:,1]"].min(), data["pos_z_recoJShower"].max(), data["E.trks.dir.z[:,1]"].max(), save_dir+"/2d_hists/", "neutrinos")
    # # histogram_2d(muons, "pos_z_recoJGandalf", "E.trks.dir.z[:,0]", False, False, data["pos_z_recoJGandalf"].min(), data["E.trks.dir.z[:,0]"].min(), data["pos_z_recoJGandalf"].max(), data["E.trks.dir.z[:,0]"].max(), save_dir+"/2d_hists/", "muons")
    # # histogram_2d(neutrinos, "pos_z_recoJGandalf", "E.trks.dir.z[:,0]", False, False, data["pos_z_recoJGandalf"].min(), data["E.trks.dir.z[:,0]"].min(), data["pos_z_recoJGandalf"].max(), data["E.trks.dir.z[:,0]"].max(), save_dir+"/2d_hists/", "neutrinos")

    # # histogram_2d(muons, "pos_x_recoJShower", "pos_y_recoJShower", False, False, data["pos_x_recoJShower"].min(), data["pos_y_recoJShower"].min(), data["pos_x_recoJShower"].max(), data["pos_y_recoJShower"].max(), save_dir+"/2d_hists/", "muons")
    # # histogram_2d(neutrinos, "pos_x_recoJShower", "pos_y_recoJShower", False, False, data["pos_x_recoJShower"].min(), data["pos_y_recoJShower"].min(), data["pos_x_recoJShower"].max(), data["pos_y_recoJShower"].max(), save_dir+"/2d_hists/", "neutrinos")