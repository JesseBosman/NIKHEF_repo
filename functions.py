import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
from tqdm import tqdm
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def list_of_ordered_features_basemodel(event_type):

    # Define the file name
    filename = '/data/antares/users/jwbosman/NIKHEF_repo/feature_ranking_basemodel_{}.txt'.format(event_type)

    # Initialize an empty list to hold the features
    features = []
    # importances = []

    # Open and read the file
    with open(filename, 'r') as file:
        # Iterate over each line in the file
        for line in file:
            # Check if the line contains a feature (starts with a number followed by a dot)
            if line.strip().startswith(tuple(f'{i}.' for i in range(1, 133))):
                # Split the line to get the feature part (splitting by space and taking the third part)
                # and the importance part (splitting by space and taking the last part)
                splitted_line = line.split()
                feature = splitted_line[2]
                # importance = splitted_line[-1]
                
                # Append the feature to the list
                features.append(feature)
                # change importance to float and append to the list
                # importance = importance.replace("(", "")
                # importance = importance.replace(")", "")
                # importance = float(importance)
                # importances.append(importance)


    # Print the list of features
    
    return features #, importances


def overview_data_table(df):
    """
    Creates a tabular overview of the data. The table includes the amount of datapoints 
    and real events (according to weight) for each different particle type.
    """
    unique_pdgids = np.unique(df["pdgid"])
    # try:
    #     unique_interactions = np.unique(df["T_sum_mc_nu.cc"])
    #     interaction_column = "T_sum_mc_nu.cc"
    # except:
    unique_interactions = np.unique(df["is_cc"])
    interaction_column = "is_cc"

    names = {13: "muon", 12: "nue", -12: "anue", 14: "numu", -14: "anumu", 16: "nutau", -16: "anutau"}

    overview = []
    n_tot_neutrinos = np.sum(df[df.is_neutrino == 1]["weight"])
    for pdgid in unique_pdgids:
        for interaction in unique_interactions:
            if pdgid == 13:
                interaction_label = ""
            else:
                if interaction == 0:
                    interaction_label = "NC"
                elif interaction == 1:
                    interaction_label = "CC"
                else:
                    interaction_label = "unknown"

            data_interaction = df[(df["pdgid"] == pdgid) & (df[interaction_column] == interaction)]
            if len(data_interaction) == 0:
                continue

            n_datapoints = len(data_interaction)

            n_events = np.sum(data_interaction["weight"])

            unique_exposures = data_interaction["exposure"].unique()

            if len(unique_exposures) > 1:
                exposure = np.sum(unique_exposures) / (365 * 24 * 3600)
            else:
                exposure = data_interaction["exposure"].mean() / (365 * 24 * 3600)

            percentage = n_events / n_tot_neutrinos * 100

            overview.append({"type": names[pdgid], "interaction": interaction_label, r"Signal ($\%$)": np.format_float_positional(percentage,1), r"$\#$ datapoints": n_datapoints, r"$\#$ real events": np.format_float_positional(n_events, 1), "exposure (yr)": np.format_float_positional(exposure, 1)})



    overview = pd.DataFrame(overview)

    print(overview.to_latex(index=False))

    return overview

def remove_infs(df):
    """
    Removes the Inf entries fromt the dataframe. Replace the (-)Inf values with the (minimum)maximum value of the column.
    """
    for column in df.columns:
        n_pos_inf = np.sum(df[column] == np.inf)
        n_neg_inf = np.sum(df[column] == -np.inf)
        if n_pos_inf > 0:
            cond = df[column] == np.inf
            col_max = np.max(df[column][np.isfinite(df[column])])
            if col_max <0:
                replacement = 0
            
            else:
                replacement = 1.5*col_max
            df[column] = df[column].mask(cond, other = replacement, inplace = False)
            print("Replaced {} inf values in column {} with value {}.".format(n_pos_inf, column, replacement))
        elif n_neg_inf > 0:
            cond = df[column] == -np.inf
            col_min = np.min(df[column][np.isfinite(df[column])])
            if col_min < 0:
                replacement = 1.5*col_min
            else:
                replacement = 0
            df[column] = df[column].mask(cond, other = replacement, inplace = False)
            print("Replaced {} -inf values in column {} with value {}.".format(n_neg_inf, column, replacement))
        else:
            continue
    
    return df

def muon_contamination(muons, neutrs):
    
    score_thresholds = np.logspace(0.001,1,1000)
    muon_contamination_percs = []


    for threshold in score_thresholds:
        
        rem_neutrs = np.sum(neutrs[neutrs['muon_score']<=threshold]["flux_weight"])
        rem_muons = np.sum(muons[muons['muon_score']<=threshold]["flux_weight"])
        muon_contamination_perc = rem_muons/(rem_muons+rem_neutrs)*100
        muon_contamination_percs.append(muon_contamination_perc)

    return np.array(muon_contamination_percs)

def neutrino_efficiency(neutrs):

    score_thresholds = np.logspace(0.001,1,1000)
    neutrino_efficiency_percs = []

    for threshold in score_thresholds:
        
        rem_neutrs = np.sum(neutrs[neutrs['muon_score']<=threshold]["flux_weight"])
        neutrino_efficiency_perc = rem_neutrs/np.sum(neutrs["flux_weight"])*100
        neutrino_efficiency_percs.append(neutrino_efficiency_perc)

    return np.array(neutrino_efficiency_percs)

def direction_resolution(df):
    """ Calculate the resolution of the direction
        Resolution is defined as the angle between the true direction and the predicted direction
        Resolution is calculated as the dot product between the true direction and the predicted direction
        The dot product is then converted to an angle using the arccos function
        The resolution is then calculated as the mean of the arccos values
        The resolution is then converted to degrees"""

    resolution = np.arccos((df['dir_x']*df['jmuon_dir_x'] + df['dir_y']*df['jmuon_dir_y'] + df['dir_z']*df['jmuon_dir_z']))

    return resolution


def remove_nans(df, threshold = 0.2):
    """
    Removes the nan entries from the dataframe. If the amount of columns with nan values is greater than 1000,
     the column is dropped. Else, if the amount of nan values in a column is greater than 0, the row is dropped.
    """
    for column in df.columns:
        column_isna = df[column].isna()
        n_nan = np.sum(column_isna)

        if n_nan/len(df) > threshold:
            df = df.drop(columns=[column])
            print("Dropped column: {}".format(column))

        elif n_nan > 0:
            indices = df[df[column].isna()].index
            df = df.drop(indices)
            print("Dropped {} rows with nan values in column: {}".format(n_nan,column))
        
        else:
            continue
    
    return df

def load_data(detector, type = "all"):
    """
    Loads the datafiles in the directory and returns the concatonated dataframe.

    Args:
        directory: directory with the datafiles
        type: type of the datafiles. Options: 'muon', 'neutrino', 'noise', 'all'. Default: 'all'
    """

    directory = "/data/antares/users/jwbosman/{}/data/".format(detector)

    df = pd.DataFrame()
    if type == "all":
        for path in os.listdir(directory):
            new_df = pd.read_hdf(directory + path)
            if "mupage" in path:
               new_df["T.sum_mc_nu.cc"] = np.float32(0)
            df = pd.concat([df, new_df], ignore_index=True)
        
    elif type == "neutrino":
        for path in os.listdir(directory):
            if "gsg" in path:
                df = pd.concat([df, pd.read_hdf(directory + path)], ignore_index=True)
    elif type == "muon":
        for path in os.listdir(directory):
            if "mupage" in path:
                df = pd.concat([df, pd.read_hdf(directory + path)], ignore_index=True)
                df["T.sum_mc_nu.cc"] = np.float32(0)
        
    elif type == "noise":
        for path in os.listdir(directory):
            if "noise" in path:
                df = pd.concat([df, pd.read_hdf(directory + path)], ignore_index=True)
    else:
        raise ValueError("Invalid type. Options: 'muon', 'neutrino', 'noise', 'all'")

    return df

def sample_data(df_muon, df_neutr, random_state = 42):
    """
    Samples the data to have the same amount of muons and neutrinos.
    
    Returns:
        df_atm_muon: dataframe with the atm_muon data
        df_atm_neutr: dataframe with the atm_neutrino data
    """
    min_samples = min(len(df_muon), len(df_neutr))
    df_muon = df_muon.sample(n=min_samples, random_state=random_state)
    df_neutr = df_neutr.sample(n=min_samples,random_state=random_state)
    
    return df_muon, df_neutr


def determine_zenith(X_data):
    dir_vectors = X_data[["jmuon_dir_x", "jmuon_dir_y", "jmuon_dir_z"]].to_numpy()
    xy_plane_vectors = np.copy(dir_vectors)
    xy_plane_vectors[:,2] = 0
    xy_plane_vectors = xy_plane_vectors/np.linalg.norm(xy_plane_vectors, axis=1)[:, np.newaxis]


    # Calculate dot product for each pair of vectors
    dot_products = np.einsum('ij,ij->i', dir_vectors, xy_plane_vectors)

    # Calculate zeniths using arccos
    zeniths = np.arccos(dot_products)

    # Adjust zeniths based on z coordinate
    zeniths[dir_vectors[:,2] < 0] *= -1

    return zeniths

def determine_bins(X_data, n_bins):
    bins = []
    for column in X_data.columns:
        bins.append(np.histogram_bin_edges(X_data[column], bins=n_bins))
    
    return np.array(bins).reshape(len(X_data.columns), n_bins+1)

def plot_confusion_matrix(y_test, X_test, clf):

    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    # Visualize confusion matrix using seaborn heatmap
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.title("Confusion Matrix: muon = 0, neutrino = 1") 
    plt.show()

# def prediction_difficulties_types(specific_types_test, y_pred, y_test, clf):
#     counted_total = Counter(specific_types_test)
#     # y_pred = clf.predict(X_test)
#     false = np.where(y_pred != y_test)[0]
#     wrong_prediction_types = specific_types_test.iloc[false]

#     counted_wrong = Counter(wrong_prediction_types)
#     for key in counted_total.keys():
#         counted_wrong[key] = counted_wrong[key]/counted_total[key]
    
#     df = pd.DataFrame.from_dict(counted_wrong, 'index').transpose()

#     return df
def zoomed_in_hist(data, column, save_dir, width, height, xlims, ylims, log_parent = False, log_child = False):
    muons = data[data["is_neutrino"]==0]
    neutrinos = data[data["is_neutrino"]==1]
    minimum = data[column].min()
    maximum = data[column].max()

    fig, ax = plt.subplots()

    
    ax.hist(neutrinos[column], bins=100, alpha=0.5, color="blue", weights=neutrinos["weight_one_year"], range=(minimum, maximum), log=log_parent, label="neutrinos", density = True)
    ax.hist(muons[column], bins=100, alpha=0.5, color="red", weights=muons["weight_one_year"], range=(minimum, maximum), log=log_parent, label="muons", density = True)

    ax.set_ylabel("Weighted and normalized number of events")
    ax.set_title(column)

    axins = inset_axes(ax, width= width, height= height, loc="upper right")  
    
    axins.hist(neutrinos[column], bins=100, alpha=0.5, color="blue", weights=neutrinos["weight_one_year"], range=xlims, log=log_child, label="neutrinos", density = True)
    axins.hist(muons[column], bins=100, alpha=0.5, color="red", weights=muons["weight_one_year"], range=xlims, log=log_child, label="muons", density = True)
    # axins.set_xlim())
    axins.legend()
    ax.legend()

    # mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5")
    
    # plt.tight_layout()
    plt.savefig(save_dir+"/1d_hists/"+column+"zoomed"+".png")
    plt.close()


def histogram_2d(data, column_1, column_2, col_1_log, col_2_log, min_1, min_2, max_1, max_2, save_dir, type = "muons"):
    title = column_1+" vs "+column_2+ " for "+type
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    save_dir+= column_1+" vs "+column_2 + type + ".png"
    if col_1_log:
        if min_1 == 0:
            min_1 = 10**-10
        bins_1 = np.logspace(np.log10(min_1), np.log10(max_1), 100)
    else:
        bins_1 = 100
    if col_2_log:
        if min_2 == 0:
            min_2 = 10**-10
        bins_2 = np.logspace(np.log10(min_2), np.log10(max_2), 100)
    else:
        bins_2 = 100


    hist = plt.hist2d(data[column_1], data[column_2], weights= data["weight_one_year"] ,bins = [bins_1, bins_2], cmap='viridis', range=[[min_1, max_1], [min_2, max_2]])
    colorbar = plt.colorbar(hist[3])
    colorbar.set_label('Weighted number of events')
    
    plt.xlabel(column_1)
    if col_1_log:
        plt.xscale("log")
    plt.ylabel(column_2)
    if col_2_log:
        plt.yscale("log")
    plt.title(title)

    # Set the font size for the tick labels

    plt.savefig(save_dir, bbox_inches = 'tight')
    plt.close()

def hist_muon_vs_neutrino_1_feature(df, column, save_dir, use_weights=False):
    neutrino_mask = df["is_neutrino"]==1
    muon_mask = df["is_neutrino"]==0
    if df[column].dtype == np.bool_:
            print(column)
            print("muons")
            print(df[muon_mask][column].value_counts())
            print("neutrinos")
            print(df[neutrino_mask][column].value_counts())
    else:
        minimum = df[column].min()
        maximum = df[column].max()
        if (column == "energy_recoJEnergy") or (column == "energy_recoJShower") or (column == "energy"):
            if use_weights:
                bins = np.logspace(np.log10(minimum), np.log10(maximum), 100)
                plt.hist(df[neutrino_mask][column], bins=bins, alpha=0.5, color="blue", label="neutrinos", log = True, density = True, weights = df[neutrino_mask]["weight_one_year"])
                plt.hist(df[muon_mask][column], bins=bins, alpha=0.5, color="red", label="muons", log = True, density = True, weights = df[muon_mask]["weight_one_year"])
                # hist_neutrinos = df[neutrino_mask][column].hist(bins=100, range = (minimum, maximum) ,alpha=0.5, color="blue", label="neutrinos", log = False, density = True, weights = df[neutrino_mask]["weight_one_year"])
                # hist_muons = df[muon_mask][column].hist(bins=100, range = (minimum, maximum) ,alpha=0.5, color="red", label="muons", log = False, density = True, weights = df[muon_mask]["weight_one_year"])
                plt.xscale("log")
                plt.ylabel("Weighted and normalized number of events")
            else:
                bins = np.logspace(np.log10(minimum), np.log10(maximum), 100)
                plt.hist(df[neutrino_mask][column], bins=bins, alpha=0.5, color="blue", label="neutrinos", log = True, density = True)
                plt.hist(df[muon_mask][column], bins=bins, alpha=0.5, color="red", label="muons", log = True, density = True)
                # hist_neutrinos = df[neutrino_mask][column].hist(bins=100, range = (minimum, maximum) , alpha=0.5, color="blue", label="neutrinos", log = False, density = True)
                # hist_muons = df[muon_mask][column].hist(bins=100, range = (minimum, maximum) , alpha=0.5, color="red", label="muons", log = False, density = True)
                plt.xscale("log")
                plt.ylabel("Normalized number of events")
            
        else:
            if use_weights:

                hist_neutrinos = df[neutrino_mask][column].hist(bins=100, range = (minimum, maximum) , alpha=0.5, color="blue", label="neutrinos", density = True, weights = df[neutrino_mask]["weight_one_year"])
                hist_muons = df[muon_mask][column].hist(bins=100, range = (minimum, maximum) ,alpha=0.5, color="red", label="muons", density = True, weights = df[muon_mask]["weight_one_year"])
                plt.ylabel("Weighted and normalized number of events")
            else:

                hist_neutrinos = df[neutrino_mask][column].hist(bins=100, range = (minimum, maximum) ,alpha=0.5, color="blue", label="neutrinos", density = True)
                hist_muons = df[muon_mask][column].hist(bins=100, range = (minimum, maximum) ,alpha=0.5, color="red", label="muons", density = True)
                plt.ylabel("Normalized number of events")
        # fig = hist.get_figure()
        plt.legend()
        plt.title(column)
        if use_weights:
            plt.savefig(save_dir+column+"weighted"+".png", bbox_inches = 'tight')
        else:
            plt.savefig(save_dir+column+".png", bbox_inches = 'tight')
        plt.close()

def histogram_muon_vs_neutrino(df, save_dir, use_weights=False):
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
            minimum = df[column].min()
            maximum = df[column].max()
            if (column == "energy_recoJEnergy") or (column == "energy_recoJShower") or (column == "energy"):
                if use_weights:

                    hist_neutrinos = df[neutrino_mask][column].hist(bins=100, range = (minimum, maximum) ,alpha=0.5, color="blue", label="neutrinos", log = False, density = True, weights = df[neutrino_mask]["weight_one_year"])
                    hist_muons = df[muon_mask][column].hist(bins=100, range = (minimum, maximum) ,alpha=0.5, color="red", label="muons", log = False, density = True, weights = df[muon_mask]["weight_one_year"])
                    plt.xscale("log")
                    plt.ylabel("Weighted and normalized number of events")
                else:

                    hist_neutrinos = df[neutrino_mask][column].hist(bins=100, range = (minimum, maximum) , alpha=0.5, color="blue", label="neutrinos", log = False, density = True)
                    hist_muons = df[muon_mask][column].hist(bins=100, range = (minimum, maximum) , alpha=0.5, color="red", label="muons", log = False, density = True)
                    plt.xscale("log")
                    plt.ylabel("Normalized number of events")
                
            else:
                if use_weights:

                    hist_neutrinos = df[neutrino_mask][column].hist(bins=100, range = (minimum, maximum) , alpha=0.5, color="blue", label="neutrinos", density = True, weights = df[neutrino_mask]["weight_one_year"])
                    hist_muons = df[muon_mask][column].hist(bins=100, range = (minimum, maximum) ,alpha=0.5, color="red", label="muons", density = True, weights = df[muon_mask]["weight_one_year"])
                    plt.ylabel("Weighted and normalized number of events")
                else:

                    hist_neutrinos = df[neutrino_mask][column].hist(bins=100, range = (minimum, maximum) ,alpha=0.5, color="blue", label="neutrinos", density = True)
                    hist_muons = df[muon_mask][column].hist(bins=100, range = (minimum, maximum) ,alpha=0.5, color="red", label="muons", density = True)
                    plt.ylabel("Normalized number of events")
            # fig = hist.get_figure()
            plt.legend()
            plt.title(column)
            if use_weights:
                plt.savefig(save_dir+column+"weighted"+".png", bbox_inches = 'tight')
            else:
                plt.savefig(save_dir+column+".png", bbox_inches = 'tight')
            plt.close()




def scatter_4d(df, column_1, column_2, column_3, column_4, save_dir, label_name, elev, azim):
    save_dir+="{}{}{}{}{}{}/".format(column_1, column_2, column_3, column_4, elev, azim)
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
    fig.savefig(fname = save_dir+label_name+ ".png", bbox_inches = 'tight')
    plt.close()
    

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
    
    score_thresholds = np.logspace(start = start ,stop = stop,num= n, base = 10)
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

def plot_efficiency_contamination(old_contamination, new_contamination, old_efficiency, new_efficiency, save_dir, event_type, train_or_test):

    fig, ax1 = plt.subplots()

    # Plot neutrino efficiency
    ax1.plot(old_contamination, old_efficiency, label='old', color= "red")
    ax1.plot(new_contamination, new_efficiency, label='new', color= "blue")
    ax1.set_xlabel('Muon Contamination [%]')
    if train_or_test == "test":
        if event_type == "track":
            ax1.axvline(x=2.66, color='black', linestyle='--', label = "threshold")

        elif event_type == "shower":
            ax1.axvline(x=2.08, color='black', linestyle='--', label = "threshold")    
    elif train_or_test == "train":
        if event_type == "track":
            ax1.axvline(x=1.70, color='black', linestyle='--', label = "threshold")

        elif event_type == "shower":
            ax1.axvline(x=1.42, color='black', linestyle='--', label = "threshold")

    ax1.set_xlim(0,4)
    ax1.set_ylim(min(old_efficiency),100)


    ax1.set_ylabel('Neutrino Efficiency [%]')
    plt.legend(loc = "best") 

    plt.title('Contamination vs Efficiency')
    plt.grid(True)
    plt.savefig(save_dir+"contamination_vs_efficiency.png", bbox_inches = 'tight')
    plt.close()




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

def figure_of_merit(exp_data, start, stop, n, new_score = False, muon_contamination_target = 0.0):
    old_neutrino_efficiency = 0
    old_threshold = start
    
    for threshold in  np.logspace(start = start ,stop = stop,num= n, base = 10):
        muon_contamination , neutrino_efficiency = get_cont_and_eff_at_threshold(exp_data, threshold, new_score)
        if muon_contamination > muon_contamination_target:
            break
        else:
            old_neutrino_efficiency = neutrino_efficiency
            old_threshold = threshold
    
    print("Muon score threshold was: ", old_threshold)
    return old_neutrino_efficiency, old_threshold


def table_of_merit(exp_data, score_threshold_LP=None, score_threshold_HP=None, event_type=None, new_score = False, save_dir= ""):
    # energy_reco_bins = [[2.0, 4.0, 4.61340151, 5.32086836, 6.13682553, 7.07791004, 8.1633102, 9.4151569, 10.85897475, 12.52420262, 14.4447938, 16.65990837, 19.21471159, 22.16129485, 25.55973776, 29.47933316, 34.0, 50, 100, 1000]]
    # cost_reco_bins =  [-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0]
    if new_score:
        score_column = "muonscore_new"
    else:
        score_column = "muonscore"

    unique_pdgids = np.unique(exp_data["pdgid"])
    # try:
    #     test = np.unique(exp_data["T_sum_mc_nu.cc"])
    #     interaction_column = "T_sum_mc_nu.cc"
    # except:
    # test = np.unique(exp_data["is_cc"])
    interaction_column = "is_cc"

    names = {13: "muon", 12: "nue", -12: "anue", 14: "numu", -14: "anumu", 16: "nutau", -16: "anutau"}

    table = []
    for pdgid in unique_pdgids:
        if pdgid == 13:
            continue
        unique_interactions = np.unique(exp_data[exp_data["pdgid"] == pdgid][interaction_column])
        for interaction in unique_interactions:

            if interaction == 0:
                interaction_label = "NC"
            elif interaction == 1:
                interaction_label = "CC"
            else:
                interaction_label = "unknown"
            data_interaction = exp_data[((exp_data["pdgid"] == pdgid) & (exp_data[interaction_column] == interaction))]
            if event_type == "track":
                data_LP = data_interaction[(data_interaction[score_column] <= score_threshold_LP)]
                data_HP = data_interaction[data_interaction[score_column] < score_threshold_HP]
                efficiency_LP = np.sum(data_LP["weight_one_year"])/np.sum(data_interaction["weight_one_year"])*100
                efficiency_HP = np.sum(data_HP["weight_one_year"])/np.sum(data_interaction["weight_one_year"])*100
                table.append({"type": names[pdgid], "interaction": interaction_label, r"$E_{\nu}$ HP": np.format_float_positional(efficiency_HP,1), r"$E_{\nu}$ LP": np.format_float_positional(efficiency_LP,1)})
            elif event_type == "shower":
                data_shower = data_interaction[(data_interaction[score_column] <= score_threshold_LP)]
                efficiency = np.sum(data_shower["weight_one_year"])/np.sum(data_interaction["weight_one_year"])*100
                table.append({"type": names[pdgid], "interaction": interaction_label, r"$E_{\nu} (\%)$": np.format_float_positional(efficiency, 1)})



            # table.append({"type": names[pdgid], "interaction": interaction_label, r"$C_{\mu}$": np.format_float_positional(contamination), r"$E_{\nu}$": np.format_float_positional(efficiency)})

    table = pd.DataFrame(table)
    table.fillna(0., inplace=True)
    path = save_dir+"table_of_merit.csv"
    table.to_csv(path_or_buf= path)
    print(r"\begin{table}[]")
    print(r"\centering")
    print(table.to_latex(index=False))
    print(r"\caption{"+f"Neutrino efficiency for different particle types and interactions for {event_type} events."+"}")
    print(r"\label{tab:"+f" table of merit {event_type}"+"}")
    print(r"\end{table}")

    
    
def get_cont_and_eff_at_threshold(data, threshold, new_score = True):
    muons = data[data["is_neutrino"]==0]
    neutrinos = data[data["is_neutrino"]==1]

    if new_score:
        column = "muonscore_new"
    else:
        column = "muonscore"

    muons_below_threshold = muons[muons[column]<=threshold]
    neutrinos_below_threshold = neutrinos[neutrinos[column]<=threshold]
    # neutrinos_below_threshold = neutrinos[neutrinos[column]<threshold] # for high purity tracks

    muon_contamination = muons_below_threshold["weight_one_year"].sum()/(muons_below_threshold["weight_one_year"].sum()+neutrinos_below_threshold["weight_one_year"].sum())*100
    neutrino_efficiency = neutrinos_below_threshold["weight_one_year"].sum()/(neutrinos["weight_one_year"].sum())*100
    return muon_contamination, neutrino_efficiency

def correlation_figure(data, features, save_dir, title = "Correlation matrix"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    data = data[features]
    corr = data.corr()
    # plt.figure(figsize=(15, 15))
    plt.matshow(corr, fignum=1)
    # plt.xticks(range(len(corr.columns)), corr.columns, rotation='vertical')
    plt.xticks(range(len(corr.columns)), [str(i) for i in range(len(corr.columns))])
    plt.yticks(range(len(corr.columns)), [str(i)+". "+column for i, column in enumerate(corr.columns)])
    plt.colorbar()
    plt.savefig(save_dir+"correlation_matrix.png", bbox_inches = 'tight')
    # plt.show()
    plt.close()

def plot_muonscore_vs_energy(data, save_dir, new_score = False):
    if new_score:
        score_column = "muonscore_new"
    else:
        score_column = "muonscore"
    plt.figure()
    plt.scatter(data["energy"], data[score_column], s = 1)
    plt.xlabel("Energy [GeV]")
    plt.ylabel("Muon score")
    plt.title("Muon score vs Energy")
    plt.savefig(save_dir+"muonscore_vs_energy.png", bbox_inches = 'tight')
    plt.close()