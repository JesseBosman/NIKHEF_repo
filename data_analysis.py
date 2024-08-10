import pandas as pd
import numpy as np
from functions import load_data, remove_nans, remove_infs, overview_data_table
import matplotlib.pyplot as plt
import os
import seaborn as sns
from tqdm import tqdm


def histogram_2d(data, column_1, column_2, col_1_log, col_2_log, min_1, min_2, max_1, max_2, save_path, title):

    hist = plt.hist2d(data[column_1], data[column_2], bins=100, cmap='viridis', range=[[min_1, max_1], [min_2, max_2]])
    colorbar = plt.colorbar(hist[3])
    colorbar.set_label('Number of events')
    plt.xlabel(column_1)
    if col_1_log:
        plt.xscale("log")
    plt.ylabel(column_2)
    if col_2_log:
        plt.yscale("log")
    plt.title(title)

    # Set the font size for the tick labels

    plt.savefig(save_path, bbox_inches = 'tight')

def histogram_muon_vs_neutrino(df, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    neutrino_mask = df["is_neutrino"]==1
    muon_mask = df["is_neutrino"]==0
    for column in tqdm(df.columns, desc="Looping over columns in histogram_muon_vs_neutrino"):
        if df[column].dtype == np.bool_:
            print(column)
            print("muons")
            print(df[muon_mask][column].value_counts())
            print("neutrinos")
            print(df[neutrino_mask][column].value_counts())
        else:
            if "energy" in column:
                hist_neutrinos = df[neutrino_mask][column].hist(bins=100, alpha=0.5, color="blue", label="neutrinos", log = True, density = True)
                hist_muons = df[muon_mask][column].hist(bins=100, alpha=0.5, color="red", label="muons", log = True, density = True)
                
            else:
                hist_neutrinos = df[neutrino_mask][column].hist(bins=100, alpha=0.5, color="blue", label="neutrinos", density = True)
                hist_muons = df[muon_mask][column].hist(bins=100, alpha=0.5, color="red", label="muons", density = True)
            # fig = hist.get_figure()
            plt.legend()
            plt.title(column)
            plt.savefig(save_dir+column+".png", bbox_inches = 'tight')
            plt.close()




def scatter_4d(df, column_1, column_2, column_3, column_4, save_dir, label_name, elev, azim):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    # fig.tight_layout(pad=-10)

    # Set the color based on the dir_z variable using a color map
    colors = df[column_4]
    # Create the scatter plot with color mapping
    scatter = ax.scatter3D(df[column_1], df[column_2], df[column_3], c=colors, cmap='coolwarm')

    # Add a colorbar
    colorbar = plt.colorbar(scatter)
    colorbar.set_label(column_4)
    # colorbar.ax.tick_params(labelsize)

    # Set labels for the axes
    ax.set_xlabel(column_1)#, fontsize = font_size_label, labelpad = pad)
    ax.set_ylabel(column_2)#, fontsize = font_size_label, labelpad = pad)
    ax.set_zlabel(column_3)#, fontsize = font_size_ladbel, labelpad = pad)


    ax.tick_params(axis='both', which='major')#, labelsize=font_size_tick)
    ax.view_init(elev=elev, azim=azim)

    plt.title(label_name)#, fontsize=font_size_title)

    # Show the plot
    plt.show()
    fig.savefig(fname = save_dir+label_name.split(" ")[0]+"{}{}{}{}elev{}azim{}.png".format(column_1, column_2, column_3, column_4, elev, azim), bbox_inches = 'tight')

def get_cont_and_eff_at_threshold(data, threshold, new_score = True):
    muons = data[data["is_neutrino"]==0]
    neutrinos = data[data["is_neutrino"]==1]

    if new_score:
        column = "muonscore_new"
    else:
        column = "muonscore"

    muons_below_threshold = muons[muons[column]<=threshold]
    neutrinos_below_threshold = neutrinos[neutrinos[column]<=threshold]

    muon_contamination = muons_below_threshold["weight_one_year"].sum()/(muons_below_threshold["weight_one_year"].sum()+neutrinos_below_threshold["weight_one_year"].sum())
    neutrino_efficiency = neutrinos_below_threshold["weight_one_year"].sum()/(neutrinos["weight_one_year"].sum())
    return muon_contamination, neutrino_efficiency

if __name__ == "__main__":
    det = "ORCA10"
    data = pd.read_hdf("/data/antares/users/jwbosman/{}/post_general_selection_cuts_data.h5".format(det))

    overview_data_table(data)




                