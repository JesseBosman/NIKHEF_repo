import joblib
import optuna
import os
# Load the study object
for file in os.listdir('.'):
    if file.endswith('.pkl'):
        study = joblib.load(file)
        file_name = file[:-4]
        # save study object as rdb
        
        storage = optuna.storages.RDBStorage(
            url="sqlite:///{}.db".format(file_name)
        )
        study_id = storage.create_new_study(study_name=file_name, directions=[optuna.study.StudyDirection.MAXIMIZE])

        for trial in study.get_trials():
            storage.create_new_trial(study_id=study_id, template_trial=trial)
