import pandas as pd
import numpy as np
from functions import load_data, remove_nans
import matplotlib.pyplot as plt
import os
import seaborn as sns


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

    plt.savefig(save_path)

def histogram_muon_vs_neutrino(df, save_dir):
    neutrino_mask = df["is_neutrino"]==1
    muon_mask = df["is_neutrino"]==0
    for column in df.columns:
        if df[column].dtype == np.bool_:
            print(column)
            print("muons")
            print(df[muon_mask][column].value_counts())
            print("neutrinos")
            print(df[neutrino_mask][column].value_counts())
        else:
            if column == "energy" or column == "E.trks.E[:,0]" or column == "E.trks.E[:,1]":
                hist_neutrinos = df[neutrino_mask][column].hist(bins=100, alpha=0.5, color="blue", label="neutrinos", log = True, density = True)
                hist_muons = df[muon_mask][column].hist(bins=100, alpha=0.5, color="red", label="muons", log = True, density = True)
                
            else:
                hist_neutrinos = df[neutrino_mask][column].hist(bins=100, alpha=0.5, color="blue", label="neutrinos", density = True)
                hist_muons = df[muon_mask][column].hist(bins=100, alpha=0.5, color="red", label="muons", density = True)
            # fig = hist.get_figure()
            plt.legend()
            plt.title(column)
            plt.savefig(save_dir+column+".png")




def scatter_4d(df, column_1, column_2, column_3, column_4, save_path, label_name):
        
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    # fig.tight_layout(pad=-10)

    # Set the color based on the dir_z variable using a color map
    colors = df[column_4]
    # Create the scatter plot with color mapping
    scatter = ax.scatter3D(df[column_1], df[column_2], df[column_3], c=colors, cmap='coolwarm')

    # Add a colorbar
    colorbar = plt.colorbar(scatter)
    colorbar.set_label(column_4, fontsize=font_size_label)
    colorbar.ax.tick_params(labelsize=font_size_tick)

    # Set labels for the axes
    ax.set_xlabel(column_1, fontsize = font_size_label, labelpad = pad)
    ax.set_ylabel(column_2, fontsize = font_size_label, labelpad = pad)
    ax.set_zlabel(column_3, fontsize = font_size_label, labelpad = pad)


    ax.tick_params(axis='both', which='major', labelsize=font_size_tick)
    ax.view_init(elev=elev, azim=azim)

    plt.title(label_name, fontsize=font_size_title)

    # Show the plot
    plt.show()
    fig.savefig(fname = "figures_classification/figures_E_less_20/upgoing/3d_scatter_"+label_name.split(" ")[0]+"{}{}{}{}elev{}azim{}.png".format(column_1, column_2, column_3, column_4, elev, azim))



if __name__ == "__main__":
    plt.rcParams["figure.figsize"] = (7,7)
    plt.rcParams["figure.labelsize"] = 14
    plt.rcParams["figure.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 12

    # data = load_data("/data/antares/users/jwbosman/ORCA10/data/", 'all')
    # for column in data.columns:
    #     print(column)
    #     for col2 in data.columns:
    #         if (column != col2) & data[column].equals(data[col2]):
    #             print(f"{column} still has duplicate {col2}")
                