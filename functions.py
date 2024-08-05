import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from collections import Counter
import os

def overview_data_table(df):
    """
    Creates a tabular overview of the data. The table includes the amount of datapoints 
    and real events (according to weight) for each different particle type.
    """
    unique_pdgids = np.unique(df["pdgid"])
    unique_interactions = np.unique(df["is_cc"])

    names = {13: "muon", 12: "nue", -12: "anue", 14: "numu", -14: "anumu", 16: "nutau", -16: "anutau"}

    overview = []
    for pdgid in unique_pdgids:
        for interaction in unique_interactions:
            if interaction == 0:
                interaction_label = "NC"
            elif interaction == 1:
                interaction_label = "CC"
            else:
                interaction_label = "unknown"

            data_interaction = df[(df["pdgid"] == pdgid) & (df["is_cc"] == interaction)]
            if len(data_interaction) == 0:
                continue

            n_datapoints = len(data_interaction)

            if pdgid == 13:
                n_events = data_interaction["weight_one_year"].sum()
            else:
                n_events = data_interaction["flux_weight"].sum()

            exposure = data_interaction["exposure"].unique()[0] / (365 * 24 * 3600)

            overview.append({"type": names[pdgid], "interaction": interaction_label, "# datapoints": n_datapoints, "# real events": np.round(n_events, 2), "exposure (yr)": np.round(exposure, 2)})

    overview = pd.DataFrame(overview)

    return overview

def remove_infs(df, threshold = 0.2):
    """
    Removes the nan entries fromt the dataframe. If the amount of columns with nan values is greater than 1000,
     the column is dropped. Else, if the amount of nan values in a column is greater than 0, the row is dropped.
    """
    for column in df.columns:
        n_inf = np.sum(df[column].isin([np.inf, -np.inf]))

        if n_inf/len(df) > threshold:
            df = df.drop(columns=[column])

        elif n_inf > 0:
            df = df.drop(df[df[column].isin([np.inf, -np.inf])].index)
        
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


def correlation_matrix(df, figsize=(15,15), plot = False):
    # Calculate the correlation matrix
    # df = df.drop(columns = [column for column in df.columns if (df[column].dtype != float) or (df[column].dtype != int)])#,"dir_x","dir_y","jmuon_dir_x","jmuon_dir_y", "jmuon_pos_x", "jmuon_pos_y", "jmuon_pos_z"])
    # print(df.head())
    corr = df.corr()
    corr = corr.round(2)


    # Plot the correlation matrix
    if plot:
        plt.figure(figsize=figsize)
        sns.heatmap(corr, annot=True)
        plt.title("Correlation matrix")
        plt.show()
    return corr

def remove_nans(df, threshold = 0.2):
    """
    Removes the nan entries from the dataframe. If the amount of columns with nan values is greater than 1000,
     the column is dropped. Else, if the amount of nan values in a column is greater than 0, the row is dropped.
    """
    for column in df.columns:
        n_nan = np.sum(df[column].isna())

        if n_nan/len(df) > threshold:
            df = df.drop(columns=[column])

        elif n_nan > 0:
            df = df.drop(df[df[column].isna()].index)
        
        else:
            continue
    
    return df

def load_data(directory, type = "all"):
    """
    Loads the datafiles in the directory and returns the concatonated dataframe.

    Args:
        directory: directory with the datafiles
        type: type of the datafiles. Options: 'muon', 'neutrino', 'noise', 'all'. Default: 'all'
    """

    df = pd.DataFrame()
    if type == "all":
        for path in os.listdir(directory):
            df = pd.concat([df, pd.read_hdf(directory + path)])
        
    elif type == "neutrino":
        for path in os.listdir(directory):
            if "gsg" in path:
                df = pd.concat([df, pd.read_hdf(directory + path)])
    elif type == "muon":
        for path in os.listdir(directory):
            if "mupage" in path:
                df = pd.concat([df, pd.read_hdf(directory + path)])
        
    elif type == "noise":
        for path in os.listdir(directory):
            if "noise" in path:
                df = pd.concat([df, pd.read_hdf(directory + path)])
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

def prediction_difficulties_types(specific_types_test, y_pred, y_test, clf):
    counted_total = Counter(specific_types_test)
    # y_pred = clf.predict(X_test)
    false = np.where(y_pred != y_test)[0]
    wrong_prediction_types = specific_types_test.iloc[false]

    counted_wrong = Counter(wrong_prediction_types)
    for key in counted_total.keys():
        counted_wrong[key] = counted_wrong[key]/counted_total[key]
    
    df = pd.DataFrame.from_dict(counted_wrong, 'index').transpose()

    return df



    

