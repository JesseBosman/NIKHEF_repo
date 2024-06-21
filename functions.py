import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from collections import Counter
from sklearn.ensemble import RandomForestClassifier as RFC
import os

def muon_contamination(muons, neutrs):
    
    score_thresholds = np.linspace(0.001,1,1000)
    muon_contamination_percs = []
    flux_correction = 1.0

    for threshold in score_thresholds:
        
        rem_neutrs = np.sum(neutrs['muon_score']<=threshold)
        rem_muons = np.sum(muons['muon_score']<=threshold)
        muon_contamination_perc = rem_muons*flux_correction/(rem_muons*flux_correction+rem_neutrs)*100
        muon_contamination_percs.append(muon_contamination_perc)

    return np.array(muon_contamination_percs)

def neutrino_efficiency(neutrs):

    score_thresholds = np.linspace(0.001,1,1000)
    neutrino_efficiency_percs = []

    for threshold in score_thresholds:
        
        rem_neutrs = np.sum(neutrs['muon_score']<=threshold)
        neutrino_efficiency_perc = rem_neutrs/len(neutrs)*100
        neutrino_efficiency_percs.append(neutrino_efficiency_perc)

    return np.array(neutrino_efficiency_percs)

def direction_resolution(df):
    # Calculate the resolution of the direction
    # Resolution is defined as the angle between the true direction and the predicted direction
    # Resolution is calculated as the dot product between the true direction and the predicted direction
    # The dot product is then converted to an angle using the arccos function
    # The resolution is then calculated as the mean of the arccos values
    # The resolution is then converted to degrees
    resolution = np.arccos((df['dir_x']*df['jmuon_dir_x'] + df['dir_y']*df['jmuon_dir_y'] + df['dir_z']*df['jmuon_dir_z']))

    return resolution

# def get_paths(directory):


# def load_data(directory):
#     """
#     Loads the v2 data for both muons and neutrinos into one df.

#     """
#     neutr_paths = os.listdir("datav2")
#     muon_path = "datav2/"+neutr_paths.pop(-2)

#     for i, path in enumerate(neutr_paths):
#         neutr_paths[i] = "datav2/"+path
#     neutr_types = ["neutrino_elec-CC","neutrino_muon-NC","neutrino_tau-CC","neutrino_anti-elec-CC","neutrino_anti-muon-CC","neutrino_anti-tau-CC","neutrino_muon-CC","neutrino_anti-muon-NC"]
#     df_atm_neutr = load_neutrino_data(neutr_paths,neutr_types)
#     # df_atm_neutr = df_atm_neutr.dropna()


#     df_atm_muon = load_muon_data(muon_path)

#     df = pd.concat([df_atm_neutr,df_atm_muon])

#     return df
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
    Removes the nan entries fromt the dataframe. If the amount of columns with nan values is greater than 1000,
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

def load_muon_data(muon_path):
    """
    Reads data from the paths to create the dataframes for the atm_muon data.
    Adds a type column to the dataframes to distinguish between the different types of neutrinos and the muons.
    
    Returns:
        df_atm_muon: dataframe with the atm_muon data
    """

    df_atm_muon = pd.read_hdf(muon_path, 'y')
    df_atm_muon['type'] = 13
    df_atm_muon["is_CC"] = 0
    
    return df_atm_muon

def load_neutrino_data(paths_atm_neutr, types_neutr, is_CC):
    """
    Reads data from the paths to create the dataframes for the atm_neutrino data.
    Adds a type column to the dataframes to distinguish between the different types of neutrinos and the muons.
    
    Returns:
        df_atm_neutr: dataframe with the atm_neutrino data
    """

    df_atm_neutr = pd.DataFrame()

    for i, path in enumerate(paths_atm_neutr):
        df = pd.read_hdf(path, "y")
        df['type'] = types_neutr[i]
        df["is_CC"] = is_CC[i]
        df_atm_neutr = pd.concat([df_atm_neutr, df])
    
    return df_atm_neutr

def preselection(df_neutr, df_muon):
    """
    Preselects the data based on the reconstruction preselection.
    """
    #selects only upgoing events
    df_neutr = df_neutr[df_neutr['jmuon_dir_z'] > 0]
    df_muon = df_muon[df_muon['jmuon_dir_z'] > 0]

    #May add extra criteria here

    return df_neutr, df_muon

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
def load_RFD_data(muon_path, neutr_paths, neutr_types, is_CC):
    """
    Reads data from the paths to create the dataframes for the atm_muon and atm_neutrino data.
    Adds a type column to the dataframes to distinguish between the different types of neutrinos and the muons.
    Adds a label column to the dataframes to distinguish between the neutrinos and the muons.
    Samples the data to have the same amount of muons and neutrinos. Numbering adheres to pdg convention.
    
    Returns:
        df_atm_muon: dataframe with the atm_muon data
        df_atm_neutr: dataframe with the atm_neutrino data
    """
    df_muon = load_muon_data(muon_path)
    df_neutr = load_neutrino_data(neutr_paths, neutr_types, is_CC)

    # df_neutr, df_muon = preselection(df_neutr, df_muon)

    df_muon = remove_nans(df_muon)
    df_neutr = remove_nans(df_neutr)

    
    df_muon = df_muon.drop(columns = [column for column in df_muon.columns if len(df_muon[column].unique()) == 1])
    df_neutr = df_neutr.drop(columns = [column for column in df_neutr.columns if len(df_neutr[column].unique()) == 1])

    # add binary labels

    df_muon['label'] = 0
    df_neutr['label'] = 1



    return df_muon, df_neutr

def data_preprocessing(paths_atm_muon, paths_atm_neutr, target_column, types_neutr, print_N=False):
    """
    Uses the load_and_sample_data function to load and sample the data.
    Combines the atm_muon and atm_neutrino dataframes.
    Drops the simulation data.
    Splits the data into features (X) and binary labels (y).

    Returns:
        X: dataframe with the features
        y: dataframe with the binary labels
    """
    
    df_atm_muon, df_atm_neutr = load_and_sample_RFD_data(paths_atm_muon, paths_atm_neutr, types_neutr, print_N)
    # Combine the dataframes
    df = (pd.concat([df_atm_muon, df_atm_neutr]))

    df.dropna(axis = 'index', inplace=True)

    # # Drop the simulation data
    df = df.drop(columns= [['energy','dir_x', 'dir_y', 'dir_z']])

    # Now your features are everything but the 'label' column
    X = df.drop(columns=['label'])
    # And your labels are the 'label' column
    y = df['label']

    # add zenith to the data
    # X['zenith'] = determine_zenith(X)

    
    return X, y

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

def feature_importance(X_train, clf):
    # Get the feature importances
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("Feature importance ranking:")
    for f in range(X_train.shape[1]):
        print(f"{f+1}. {X_train.columns[indices[f]]} ({importances[indices[f]]})")

def experiment(paths_atm_muon, paths_atm_neutr, neutr_types, n_its, n_estimators = 100, class_weight = {0: 1, 1: 1}, n_bins = 100):
    random_states = np.arange(n_its)
    test_scores = []
    df_prediction_difficulties = pd.DataFrame()
    X, y = data_preprocessing(paths_atm_muon=paths_atm_muon, paths_atm_neutr=paths_atm_neutr, types_neutr= neutr_types)
    columns = X.drop(columns = ['type']).columns

    bins = determine_bins(X.drop(columns=['type']), n_bins)
    hists_true = np.zeros((len(X.columns), n_bins), dtype = int)
    hists_false = np.zeros((len(X.columns), n_bins), dtype = int)
    
    
    for random_state in random_states:
        
        X_train_full, X_test_full, y_train, y_test = train_test_split(X, y , shuffle = True, test_size=0.2, random_state=random_state)
        
        specific_types_test = X_test_full['type']
        # specific_types_train = X_train_full['type']
        X_test = X_test_full.drop(columns=['type'])
        X_train = X_train_full.drop(columns=['type'])
        clf = RFC(n_estimators=n_estimators, random_state=random_state, class_weight=class_weight, oob_score=True, verbose = 1, n_jobs=-1, max_samples = 0.4, max_features = 0.6)
        clf.fit(X_train, y_train)
        

        test_scores.append(clf.score(X_test, y_test))
        df_prediction_difficulties = pd.concat([prediction_difficulties_types(specific_types_test, X_test, y_test, clf), df_prediction_difficulties])
    
        hists_true, hists_false = histogram_dependency_variables(X_test, y_test, clf, hists_true, hists_false, bins)

    return test_scores, df_prediction_difficulties, hists_true, hists_false, columns, bins

# def add_noise(df, noise_level):
#     df_noisy = df.copy()
#     for col in df_noisy.columns:
#         if col != 'type' and col != 'label':
#             df_noisy[col] = df_noisy[col] + np.random.normal(0, noise_level, len(df_noisy))
#     return df_noisy

def histogram_dependency_variables(X_test, y_test, y_pred, clf, hists_true, hists_false, bins):
    # y_pred = clf.predict(X_test)
    false = np.where(y_pred != y_test)[0]
    correct = np.where(y_pred == y_test)[0]
    wrong_prediction_X = X_test.iloc[false]
    correct_prediction_X = X_test.iloc[correct]

    for i, column in enumerate(X_test.columns):

        hist_true = np.histogram(correct_prediction_X[column], bins=bins[i])[0]
        hist_false = np.histogram(wrong_prediction_X[column], bins=bins[i])[0]
        
        hists_true[i] = np.add(hist_true, hists_true[i])
        hists_false[i] = np.add(hist_false, hists_false[i])    

    return hists_true, hists_false
    
    

