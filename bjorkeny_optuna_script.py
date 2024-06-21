import tensorflow as tf
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor as RFR
# from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split, KFold

# import seaborn as sns

# import os
from sklearn import preprocessing
from sklearn.decomposition import PCA
from functions import *
import keras
import optuna
import pickle

pdgid = 14
is_cc = 1
likelihood = 0.5
nan_threshold = 0.2
quantile_0 = 0.2
quantile_1 = 0.4
to_PCA = True
loss_metric = "mae"



# Load data
df_11 = pd.read_hdf("datay/new_neutrino11x_1.h5")
df_12 = pd.read_hdf("datay/new_neutrino12x_1.h5")
df_13 = pd.read_hdf("datay/new_neutrino13x_1.h5")

df = pd.concat([df_11, df_12, df_13])

# Remove nans and illegitimate values
df = remove_nans(df, nan_threshold)
df = df[np.invert((df['closest[:,0,0]']==1e20) | (df['closest[:,1,0]']==1e20) | (df['closest[:,1,2]']==1e20) | (df['closest[:,0,1]']==1e20)  )] 

# select the particle and interaction types
df = df[(df["pdgid"]==pdgid)&(df["is_cc"]==is_cc)]

# select based on energy
df = df[(df["E.trks.E[:,0]"]<=20) | (df["E.trks.E[:,1]"]<=20)]

# remove unneeded columns
df = df.drop(columns = ["exposure", "weight", "weight_rate", "E.run_id", "pdgid", "is_cc", "energy"])

# select based on likelihood

q0 = np.quantile(df["E.trks.lik[:,0]"], quantile_0)
q1 = np.quantile(df["E.trks.lik[:,1]"], 1-quantile_1)
df = df[(df["E.trks.lik[:,0]"]>q0) | (df["E.trks.lik[:,1]"] < q1)]

def objective(trial):
    # remove columns with low correlation
    df_local = df.copy()

    corr = df_local.corr()
    bjorkeny_corr = corr['T.sum_mc_nu.by']
    correlation_threshold = trial.suggest_float("correlation_threshold", 0, 0.3)
    non_rel = bjorkeny_corr.index[abs(bjorkeny_corr)<=correlation_threshold]

    df_local = df_local.drop(columns = non_rel)


    bjorkeny = df_local["T.sum_mc_nu.by"].to_numpy()
    X = df_local.drop(columns = "T.sum_mc_nu.by")

    # Scale the data
    scaler = preprocessing.StandardScaler()
    X = scaler.fit_transform(X)

    # PCA
    n_components_PCA = trial.suggest_float("n_components_PCA", 0.6, 1.0)

    pca = PCA(n_components=n_components_PCA, svd_solver="full")
    X = pca.fit_transform(X)

    # Split the data into folds
    kfold = KFold(n_splits=4, shuffle=True)

    # Train the model
    n_layers = trial.suggest_int(name = "n_layers", low = 1, high = 20, step = 1)
    n_nodes = trial.suggest_int(name = "n_nodes", low = 10, high = 100, step = 10)
    dropout_frac = trial.suggest_float("dropout_frac", 0, 0.5)
    activation_hidden = trial.suggest_categorical("activation_hidden", ["relu", "tanh", "sigmoid"])
    activation_output = trial.suggest_categorical("activation_output", ["relu", "tanh", "sigmoid"])
    learning_rate = trial.suggest_float("learning_rate", low = 1e-5, high=1e-1, log = True)

    scores = []

    for i , (train_index, test_index) in enumerate(kfold.split(X)):

        tf.keras.backend.clear_session()
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = bjorkeny[train_index], bjorkeny[test_index]

        # Train the DNN
        DNN = tf.keras.Sequential()
        DNN.add(tf.keras.layers.Input(shape=(X.shape[1],)))
        # DNN.add(tf.keras.layers.Dense(64, activation="relu"))

        if n_layers ==1:
            DNN.add(tf.keras.layers.Dense(n_nodes, activation=activation_hidden))
            DNN.add(keras.layers.Dropout(dropout_frac))
        else: 
            for _ in range(n_layers-1):
                DNN.add(tf.keras.layers.Dense(n_nodes, activation=activation_hidden))
                DNN.add(keras.layers.Dropout(dropout_frac))

            DNN.add(tf.keras.layers.Dense(int(0.5*n_nodes), activation=activation_hidden))

            DNN.add(keras.layers.Dropout(dropout_frac))

        DNN.add(tf.keras.layers.Dense(1, activation=activation_output))
        DNN.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mae")

        DNN.fit(X_train, y_train, epochs=500, batch_size=32, validation_data=(X_test, y_test), callbacks=[callback], verbose = 0)
        score = DNN.evaluate(X_test, y_test, verbose=0)
        scores.append(score)

    return np.mean(scores)
with open("bjorkeny_study_2.pkl", "rb") as f:
    study = pickle.load(f)
# study = optuna.create_study(study_name = "bjorkeny_study_2", direction="minimize")
study.optimize(objective, n_trials=30, show_progress_bar=True)

print(study.best_params)
print(study.best_value)


with open("bjorkeny_study_2.pkl", "wb") as f:
    pickle.dump(study, f)